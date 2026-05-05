#include "rescue_refit.hpp"

#include "lls.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef TCM_HAS_OPENMP
#include <omp.h>
#endif

namespace tcm {
namespace {

constexpr double kBoundHitFraction = 0.01;
constexpr double kCostImproveTolerance = 1e-12;
constexpr double kKiDropoutRatio = 0.35;
constexpr double kK3DropoutRatio = 0.35;

struct NeighborhoodSummary {
  int donor_count = 0;
  bool valid = false;
  ParameterVector params{};
  double ki = std::numeric_limits<double>::quiet_NaN();
};

std::size_t flatten_index(
    std::size_t z,
    std::size_t y,
    std::size_t x,
    std::size_t height,
    std::size_t width) {
  return (z * height + y) * width + x;
}

bool is_target_status(int32_t status) {
  return status == static_cast<int32_t>(FitStatus::Failed) ||
      status == static_cast<int32_t>(FitStatus::StepConverged) ||
      status == static_cast<int32_t>(FitStatus::MaxIterations);
}

bool is_reliable_donor_status(int32_t status) {
  return status == static_cast<int32_t>(FitStatus::GradientConverged) ||
      status == static_cast<int32_t>(FitStatus::CostConverged);
}

double median_value(std::vector<double>& values) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  if ((values.size() % 2) != 0) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1] + values[mid]);
}

double quantile(std::vector<double> values, double q) {
  if (values.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  q = std::clamp(q, 0.0, 1.0);
  std::sort(values.begin(), values.end());
  if (values.size() == 1) {
    return values.front();
  }
  const double pos = q * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(pos));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(pos));
  if (lower == upper) {
    return values[lower];
  }
  const double alpha = pos - static_cast<double>(lower);
  return (1.0 - alpha) * values[lower] + alpha * values[upper];
}

bool near_bound(double value, double lower, double upper) {
  if (!std::isfinite(value) || !std::isfinite(lower) || !std::isfinite(upper)) {
    return false;
  }
  const double span = std::max(upper - lower, 1e-8);
  const double tol = std::max(1e-6, kBoundHitFraction * span);
  return std::abs(value - lower) <= tol || std::abs(value - upper) <= tol;
}

bool active_bound_hit(const ParameterVector& params, const SolverOptions& options) {
  std::array<double, kMaxParameterCount> values{};
  pack_active(
      params,
      options.model,
      options.fit_vb,
      options.fit_delay,
      options.fit_dispersion,
      values);
  const int n = active_parameter_count(
      options.model,
      options.fit_vb,
      options.fit_delay,
      options.fit_dispersion);
  for (int i = 0; i < n; ++i) {
    if (near_bound(
            values[static_cast<std::size_t>(i)],
            options.bounds.lower[static_cast<std::size_t>(i)],
            options.bounds.upper[static_cast<std::size_t>(i)])) {
      return true;
    }
  }
  return false;
}

bool evaluate_candidate(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& tac,
    const ParameterVector& params,
    double& weighted_cost,
    double& rmse) {
  std::vector<double> predicted;
  if (!evaluate_model(options.model, params, time_minutes, aif, predicted) ||
      predicted.size() != tac.size()) {
    return false;
  }
  weighted_cost = 0.0;
  double sq = 0.0;
  for (std::size_t i = 0; i < tac.size(); ++i) {
    const double diff = predicted[i] - tac[i];
    weighted_cost += 0.5 * diff * diff * weights[i] * weights[i];
    sq += diff * diff;
  }
  rmse = std::sqrt(sq / static_cast<double>(tac.size()));
  return std::isfinite(weighted_cost) && std::isfinite(rmse);
}

ParameterVector params_from_maps(
    ModelKind model,
    std::size_t index,
    const std::vector<double>& k1_map,
    const std::vector<double>& k2_map,
    const std::vector<double>& k3_map,
    const std::vector<double>* k4_map,
    const std::vector<double>& vb_map,
    const std::vector<double>* delay_map,
    const std::vector<double>* dispersion_map) {
  ParameterVector params{};
  params.K1 = k1_map[index];
  params.k2 = k2_map[index];
  params.k3 = k3_map[index];
  params.k4 = model == ModelKind::Reversible2TCM ? (*k4_map)[index] : 0.0;
  params.vB = vb_map[index];
  params.delay = delay_map != nullptr ? (*delay_map)[index] : 0.0;
  params.dispersion = dispersion_map != nullptr ? (*dispersion_map)[index] : 0.0;
  return params;
}

bool finite_params(const ParameterVector& params, ModelKind model) {
  return std::isfinite(params.K1) && std::isfinite(params.k2) && std::isfinite(params.k3) &&
      std::isfinite(params.vB) && std::isfinite(params.delay) && std::isfinite(params.dispersion) &&
      (model != ModelKind::Reversible2TCM || std::isfinite(params.k4));
}

NeighborhoodSummary collect_neighborhood_summary(
    std::size_t center,
    const std::vector<int32_t>& spatial_mask,
    const std::vector<int32_t>& status_map,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    int radius,
    const std::vector<double>& k1_map,
    const std::vector<double>& k2_map,
    const std::vector<double>& k3_map,
    const std::vector<double>* k4_map,
    const std::vector<double>& vb_map,
    const std::vector<double>* delay_map,
    const std::vector<double>* dispersion_map,
    const std::vector<double>& ki_map,
    ModelKind model,
    int min_reliable_neighbors) {
  NeighborhoodSummary summary;
  const std::size_t plane = height * width;
  const std::size_t z = center / plane;
  const std::size_t rem = center % plane;
  const std::size_t y = rem / width;
  const std::size_t x = rem % width;

  const std::size_t z0 = z > static_cast<std::size_t>(radius) ? z - static_cast<std::size_t>(radius) : 0;
  const std::size_t y0 = y > static_cast<std::size_t>(radius) ? y - static_cast<std::size_t>(radius) : 0;
  const std::size_t x0 = x > static_cast<std::size_t>(radius) ? x - static_cast<std::size_t>(radius) : 0;
  const std::size_t z1 = std::min(depth - 1, z + static_cast<std::size_t>(radius));
  const std::size_t y1 = std::min(height - 1, y + static_cast<std::size_t>(radius));
  const std::size_t x1 = std::min(width - 1, x + static_cast<std::size_t>(radius));

  std::vector<double> k1_values;
  std::vector<double> k2_values;
  std::vector<double> k3_values;
  std::vector<double> k4_values;
  std::vector<double> vb_values;
  std::vector<double> delay_values;
  std::vector<double> dispersion_values;
  std::vector<double> ki_values;
  const std::size_t reserve = static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1) * (2 * radius + 1));
  k1_values.reserve(reserve);
  k2_values.reserve(reserve);
  k3_values.reserve(reserve);
  k4_values.reserve(reserve);
  vb_values.reserve(reserve);
  delay_values.reserve(reserve);
  dispersion_values.reserve(reserve);
  ki_values.reserve(reserve);

  for (std::size_t zz = z0; zz <= z1; ++zz) {
    for (std::size_t yy = y0; yy <= y1; ++yy) {
      for (std::size_t xx = x0; xx <= x1; ++xx) {
        const std::size_t index = flatten_index(zz, yy, xx, height, width);
        if (index == center || spatial_mask[index] == 0 || !is_reliable_donor_status(status_map[index])) {
          continue;
        }
        const ParameterVector params = params_from_maps(
            model,
            index,
            k1_map,
            k2_map,
            k3_map,
            k4_map,
            vb_map,
            delay_map,
            dispersion_map);
        if (!finite_params(params, model) || !std::isfinite(ki_map[index])) {
          continue;
        }
        k1_values.push_back(params.K1);
        k2_values.push_back(params.k2);
        k3_values.push_back(params.k3);
        if (model == ModelKind::Reversible2TCM) {
          k4_values.push_back(params.k4);
        }
        vb_values.push_back(params.vB);
        if (delay_map != nullptr) {
          delay_values.push_back(params.delay);
        }
        if (dispersion_map != nullptr) {
          dispersion_values.push_back(params.dispersion);
        }
        ki_values.push_back(ki_map[index]);
      }
    }
  }

  summary.donor_count = static_cast<int>(ki_values.size());
  if (summary.donor_count < min_reliable_neighbors) {
    return summary;
  }

  summary.params.K1 = median_value(k1_values);
  summary.params.k2 = median_value(k2_values);
  summary.params.k3 = median_value(k3_values);
  summary.params.k4 = model == ModelKind::Reversible2TCM ? median_value(k4_values) : 0.0;
  summary.params.vB = median_value(vb_values);
  summary.params.delay = delay_map != nullptr ? median_value(delay_values) : 0.0;
  summary.params.dispersion = dispersion_map != nullptr ? median_value(dispersion_values) : 0.0;
  summary.ki = median_value(ki_values);
  summary.valid = finite_params(summary.params, model) && std::isfinite(summary.ki);
  return summary;
}

ParameterVector clip_seed(const ParameterVector& seed, const SolverOptions& options) {
  return clip_to_bounds(
      seed,
      options.model,
      options.fit_vb,
      options.fixed_vb,
      options.fit_delay,
      options.fixed_delay,
      options.fit_dispersion,
      options.fixed_dispersion,
      options.bounds);
}

ParameterVector nudge_from_bounds(const ParameterVector& seed, const SolverOptions& options) {
  std::array<double, kMaxParameterCount> values{};
  pack_active(
      seed,
      options.model,
      options.fit_vb,
      options.fit_delay,
      options.fit_dispersion,
      values);
  const int n = active_parameter_count(
      options.model,
      options.fit_vb,
      options.fit_delay,
      options.fit_dispersion);
  for (int i = 0; i < n; ++i) {
    const double lower = options.bounds.lower[static_cast<std::size_t>(i)];
    const double upper = options.bounds.upper[static_cast<std::size_t>(i)];
    const double span = std::max(upper - lower, 1e-8);
    const double offset = 0.05 * span;
    if (std::abs(values[static_cast<std::size_t>(i)] - lower) <= std::max(1e-6, kBoundHitFraction * span)) {
      values[static_cast<std::size_t>(i)] = std::min(upper, lower + offset);
    } else if (
        std::abs(values[static_cast<std::size_t>(i)] - upper) <= std::max(1e-6, kBoundHitFraction * span)) {
      values[static_cast<std::size_t>(i)] = std::max(lower, upper - offset);
    }
  }
  return unpack_active(
      values,
      options.model,
      options.fit_vb,
      options.fixed_vb,
      options.fit_delay,
      options.fixed_delay,
      options.fit_dispersion,
      options.fixed_dispersion);
}

ParameterVector boost_k3(
    const ParameterVector& seed,
    const NeighborhoodSummary* neighborhood,
    const SolverOptions& options) {
  ParameterVector out = seed;
  const double neighbor_k3 = (neighborhood != nullptr && neighborhood->valid) ? neighborhood->params.k3 : 0.0;
  const double k3_floor = std::max({0.02, 0.15 * std::max(seed.k2, 0.0), 0.5 * neighbor_k3});
  out.k3 = std::max(out.k3, k3_floor);
  if (options.model == ModelKind::Reversible2TCM) {
    out.k4 *= 0.5;
  }
  return clip_seed(out, options);
}

ParameterVector reduce_k4(const ParameterVector& seed, const SolverOptions& options) {
  ParameterVector out = seed;
  if (options.model == ModelKind::Reversible2TCM) {
    out.k4 *= 0.5;
  }
  return clip_seed(out, options);
}

bool similar_seed(
    const ParameterVector& a,
    const ParameterVector& b,
    ModelKind model,
    bool fit_delay,
    bool fit_dispersion) {
  const auto close = [](double x, double y) {
    return std::abs(x - y) <= 1e-6 * std::max({1.0, std::abs(x), std::abs(y)});
  };
  if (!close(a.K1, b.K1) || !close(a.k2, b.k2) || !close(a.k3, b.k3) || !close(a.vB, b.vB)) {
    return false;
  }
  if (model == ModelKind::Reversible2TCM && !close(a.k4, b.k4)) {
    return false;
  }
  if (fit_delay && !close(a.delay, b.delay)) {
    return false;
  }
  if (fit_dispersion && !close(a.dispersion, b.dispersion)) {
    return false;
  }
  return true;
}

void add_seed_if_distinct(
    std::vector<ParameterVector>& seeds,
    const ParameterVector& seed,
    const SolverOptions& options) {
  const ParameterVector clipped = clip_seed(seed, options);
  for (const ParameterVector& existing : seeds) {
    if (similar_seed(
            existing,
            clipped,
            options.model,
            options.fit_delay,
            options.fit_dispersion)) {
      return;
    }
  }
  seeds.push_back(clipped);
}

}  // namespace

RescueRefitResult apply_rescue_refit(
    const SolverOptions& solver_options,
    const RescueRefitOptions& rescue_options,
    const std::vector<int32_t>& spatial_mask,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& fit_image,
    std::vector<double>& k1_map,
    std::vector<double>& k2_map,
    std::vector<double>& k3_map,
    std::vector<double>* k4_map,
    std::vector<double>& vb_map,
    std::vector<double>* delay_map,
    std::vector<double>* dispersion_map,
    std::vector<double>& ki_map,
    std::vector<double>& rmse_map,
    std::vector<int32_t>& iter_map,
    std::vector<int32_t>& status_map) {
  const std::size_t voxel_count = depth * height * width;
  const std::size_t t_count = time_minutes.size();
  if (spatial_mask.size() != voxel_count || status_map.size() != voxel_count ||
      k1_map.size() != voxel_count || k2_map.size() != voxel_count ||
      k3_map.size() != voxel_count || vb_map.size() != voxel_count ||
      ki_map.size() != voxel_count || rmse_map.size() != voxel_count ||
      iter_map.size() != voxel_count || fit_image.size() != voxel_count * t_count) {
    throw std::runtime_error("Rescue refit input size mismatch");
  }
  if (solver_options.model == ModelKind::Reversible2TCM &&
      (k4_map == nullptr || k4_map->size() != voxel_count)) {
    throw std::runtime_error("Reversible rescue refit requires k4 map");
  }

  RescueRefitResult result;
  result.target_mask.assign(voxel_count, 0);
  result.applied_mask.assign(voxel_count, 0);
  result.unresolved_mask.assign(voxel_count, 0);

  if (!rescue_options.enable) {
    return result;
  }
  if (rescue_options.radius < 1 || rescue_options.min_reliable_neighbors < 1 || rescue_options.iterations < 1) {
    throw std::runtime_error("Invalid rescue refit options");
  }

  const std::vector<double> k1_source = k1_map;
  const std::vector<double> k2_source = k2_map;
  const std::vector<double> k3_source = k3_map;
  const std::vector<double> vb_source = vb_map;
  const std::vector<double> ki_source = ki_map;
  const std::vector<double> rmse_source = rmse_map;
  const std::vector<int32_t> iter_source = iter_map;
  const std::vector<int32_t> status_source = status_map;
  const std::vector<double> k4_source =
      solver_options.model == ModelKind::Reversible2TCM ? *k4_map : std::vector<double>{};
  const std::vector<double> delay_source = delay_map != nullptr ? *delay_map : std::vector<double>{};
  const std::vector<double> dispersion_source =
      dispersion_map != nullptr ? *dispersion_map : std::vector<double>{};

  std::vector<double> finite_rmse;
  finite_rmse.reserve(voxel_count);
  for (std::size_t i = 0; i < voxel_count; ++i) {
    if (spatial_mask[i] != 0 && std::isfinite(rmse_source[i])) {
      finite_rmse.push_back(rmse_source[i]);
    }
  }
  const double high_rmse_threshold = quantile(finite_rmse, rescue_options.high_rmse_quantile);
  const double k3_zero_threshold = std::max(1e-3, 0.005 * solver_options.bounds.upper[2]);

  std::vector<double> k1_out = k1_map;
  std::vector<double> k2_out = k2_map;
  std::vector<double> k3_out = k3_map;
  std::vector<double> vb_out = vb_map;
  std::vector<double> ki_out = ki_map;
  std::vector<double> rmse_out = rmse_map;
  std::vector<int32_t> iter_out = iter_map;
  std::vector<int32_t> status_out = status_map;
  std::vector<double> k4_out =
      solver_options.model == ModelKind::Reversible2TCM ? *k4_map : std::vector<double>{};
  std::vector<double> delay_out = delay_map != nullptr ? *delay_map : std::vector<double>{};
  std::vector<double> dispersion_out = dispersion_map != nullptr ? *dispersion_map : std::vector<double>{};

  SolverOptions rescue_solver = solver_options;
  rescue_solver.max_iterations = rescue_options.iterations;
  rescue_solver.use_linear_warm_start = false;

#ifdef TCM_HAS_OPENMP
#pragma omp parallel
#endif
  {
    std::vector<double> tac(t_count, 0.0);

#ifdef TCM_HAS_OPENMP
#pragma omp for schedule(dynamic, 128)
#endif
    for (std::int64_t voxel = 0; voxel < static_cast<std::int64_t>(voxel_count); ++voxel) {
      const std::size_t index = static_cast<std::size_t>(voxel);
      if (spatial_mask[index] == 0) {
        continue;
      }

      const NeighborhoodSummary neighborhood = collect_neighborhood_summary(
          index,
          spatial_mask,
          status_source,
          depth,
          height,
          width,
          rescue_options.radius,
          k1_source,
          k2_source,
          k3_source,
          solver_options.model == ModelKind::Reversible2TCM ? &k4_source : nullptr,
          vb_source,
          delay_map != nullptr ? &delay_source : nullptr,
          dispersion_map != nullptr ? &dispersion_source : nullptr,
          ki_source,
          solver_options.model,
          rescue_options.min_reliable_neighbors);

      const ParameterVector current_params = params_from_maps(
          solver_options.model,
          index,
          k1_source,
          k2_source,
          k3_source,
          solver_options.model == ModelKind::Reversible2TCM ? &k4_source : nullptr,
          vb_source,
          delay_map != nullptr ? &delay_source : nullptr,
          dispersion_map != nullptr ? &dispersion_source : nullptr);
      const bool current_valid = finite_params(current_params, solver_options.model);
      const bool bound_hit = current_valid && active_bound_hit(current_params, solver_options);
      const bool high_rmse = std::isfinite(rmse_source[index]) && rmse_source[index] >= high_rmse_threshold;
      bool local_dropout = false;
      if (neighborhood.valid && neighborhood.ki > 1e-6 && std::isfinite(ki_source[index]) && current_valid) {
        const bool ki_drop = ki_source[index] < kKiDropoutRatio * neighborhood.ki;
        const bool k3_drop =
            current_params.k3 <= k3_zero_threshold || current_params.k3 < kK3DropoutRatio * neighborhood.params.k3;
        local_dropout = ki_drop && k3_drop;
      }

      const bool target =
          is_target_status(status_source[index]) || (bound_hit && high_rmse) || local_dropout;
      if (!target) {
        continue;
      }
      result.target_mask[index] = 1;

      for (std::size_t t = 0; t < t_count; ++t) {
        tac[t] = fit_image[t * voxel_count + index];
      }

      double best_cost = std::numeric_limits<double>::infinity();
      double best_rmse = rmse_source[index];
      ParameterVector best_params = current_params;
      int32_t best_status = status_source[index];
      int32_t best_iterations = iter_source[index];
      if (current_valid) {
        double current_cost = 0.0;
        double current_rmse = 0.0;
        if (evaluate_candidate(
                solver_options,
                time_minutes,
                aif,
                weights,
                tac,
                current_params,
                current_cost,
                current_rmse)) {
          best_cost = current_cost;
          best_rmse = current_rmse;
        }
      }

      std::vector<ParameterVector> seeds;
      if (current_valid) {
        add_seed_if_distinct(seeds, current_params, solver_options);
        add_seed_if_distinct(seeds, nudge_from_bounds(current_params, solver_options), solver_options);
        add_seed_if_distinct(seeds, boost_k3(current_params, neighborhood.valid ? &neighborhood : nullptr, solver_options), solver_options);
        if (solver_options.model == ModelKind::Reversible2TCM) {
          add_seed_if_distinct(seeds, reduce_k4(current_params, solver_options), solver_options);
        }
      }

      if (solver_options.use_linear_warm_start) {
        const WarmStartResult warm = linear_warm_start(
            solver_options.model,
            solver_options.fit_vb,
            solver_options.fixed_vb,
            solver_options.fit_delay,
            solver_options.fixed_delay,
            solver_options.fit_dispersion,
            solver_options.fixed_dispersion,
            solver_options.bounds,
            time_minutes,
            aif,
            tac);
        if (warm.valid) {
          add_seed_if_distinct(seeds, warm.params, solver_options);
          add_seed_if_distinct(
              seeds,
              boost_k3(warm.params, neighborhood.valid ? &neighborhood : nullptr, solver_options),
              solver_options);
        }
      }

      if (neighborhood.valid) {
        add_seed_if_distinct(seeds, neighborhood.params, solver_options);
        add_seed_if_distinct(
            seeds,
            boost_k3(neighborhood.params, &neighborhood, solver_options),
            solver_options);
        if (solver_options.model == ModelKind::Reversible2TCM) {
          add_seed_if_distinct(seeds, reduce_k4(neighborhood.params, solver_options), solver_options);
        }
      }

      bool improved = false;
      for (const ParameterVector& seed : seeds) {
        const FitResult fit = fit_voxel_with_initial_guess(
            rescue_solver,
            time_minutes,
            aif,
            weights,
            tac,
            seed);
        if ((fit.status == FitStatus::InvalidInput || fit.status == FitStatus::Failed) ||
            !finite_params(fit.params, solver_options.model) || !std::isfinite(fit.weighted_cost)) {
          continue;
        }
        if (fit.weighted_cost + kCostImproveTolerance < best_cost) {
          best_cost = fit.weighted_cost;
          best_rmse = fit.rmse;
          best_params = fit.params;
          best_status = static_cast<int32_t>(fit.status);
          best_iterations = iter_source[index] + fit.iterations;
          improved = true;
        }
      }

      if (!improved) {
        result.unresolved_mask[index] = 1;
        continue;
      }

      result.applied_mask[index] = 1;
      k1_out[index] = best_params.K1;
      k2_out[index] = best_params.k2;
      k3_out[index] = best_params.k3;
      vb_out[index] = best_params.vB;
      if (solver_options.model == ModelKind::Reversible2TCM) {
        k4_out[index] = best_params.k4;
      }
      if (delay_map != nullptr) {
        delay_out[index] = best_params.delay;
      }
      if (dispersion_map != nullptr) {
        dispersion_out[index] = best_params.dispersion;
      }
      ki_out[index] = compute_ki(best_params);
      rmse_out[index] = best_rmse;
      iter_out[index] = best_iterations;
      status_out[index] = best_status;
    }
  }

  result.targeted_voxels = static_cast<std::size_t>(std::count(result.target_mask.begin(), result.target_mask.end(), 1));
  result.applied_voxels = static_cast<std::size_t>(std::count(result.applied_mask.begin(), result.applied_mask.end(), 1));

  k1_map = std::move(k1_out);
  k2_map = std::move(k2_out);
  k3_map = std::move(k3_out);
  vb_map = std::move(vb_out);
  ki_map = std::move(ki_out);
  rmse_map = std::move(rmse_out);
  iter_map = std::move(iter_out);
  status_map = std::move(status_out);
  if (solver_options.model == ModelKind::Reversible2TCM) {
    *k4_map = std::move(k4_out);
  }
  if (delay_map != nullptr) {
    *delay_map = std::move(delay_out);
  }
  if (dispersion_map != nullptr) {
    *dispersion_map = std::move(dispersion_out);
  }

  return result;
}

}  // namespace tcm
