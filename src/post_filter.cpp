#include "post_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tcm {
namespace {

struct NeighborhoodPlan {
  std::size_t center = 0;
  std::vector<std::size_t> donors;
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
  return status == static_cast<int32_t>(FitStatus::StepConverged) ||
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

std::vector<double> apply_plan_to_map(
    const std::vector<double>& source,
    const std::vector<NeighborhoodPlan>& plans,
    int min_reliable_neighbors) {
  std::vector<double> filtered = source;
  std::vector<double> values;
  for (const NeighborhoodPlan& plan : plans) {
    values.clear();
    values.reserve(plan.donors.size());
    for (std::size_t donor : plan.donors) {
      const double value = source[donor];
      if (std::isfinite(value)) {
        values.push_back(value);
      }
    }
    if (static_cast<int>(values.size()) < min_reliable_neighbors) {
      continue;
    }
    filtered[plan.center] = median_value(values);
  }
  return filtered;
}

}  // namespace

PostFilterResult apply_selective_status_median_filter(
    ModelKind model,
    const PostFilterOptions& options,
    const std::vector<int32_t>& spatial_mask,
    const std::vector<int32_t>& status_map,
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    std::vector<double>& k1_map,
    std::vector<double>& k2_map,
    std::vector<double>& k3_map,
    std::vector<double>* k4_map,
    std::vector<double>& vb_map,
    std::vector<double>* delay_map,
    std::vector<double>* dispersion_map,
    std::vector<double>& ki_map) {
  const std::size_t voxel_count = depth * height * width;
  if (spatial_mask.size() != voxel_count || status_map.size() != voxel_count ||
      k1_map.size() != voxel_count || k2_map.size() != voxel_count ||
      k3_map.size() != voxel_count || vb_map.size() != voxel_count || ki_map.size() != voxel_count) {
    throw std::runtime_error("Post-filter input size mismatch");
  }
  if (model == ModelKind::Reversible2TCM && (k4_map == nullptr || k4_map->size() != voxel_count)) {
    throw std::runtime_error("Reversible post-filter requires k4 map");
  }

  PostFilterResult result;
  result.target_mask.assign(voxel_count, 0);
  result.applied_mask.assign(voxel_count, 0);
  result.unresolved_mask.assign(voxel_count, 0);

  if (!options.enable_status_median || options.radius < 1 || options.min_reliable_neighbors < 1) {
    return result;
  }

  std::vector<int32_t> donor_mask(voxel_count, 0);
  for (std::size_t index = 0; index < voxel_count; ++index) {
    if (spatial_mask[index] == 0) {
      continue;
    }
    const int32_t status = status_map[index];
    if (is_target_status(status)) {
      result.target_mask[index] = 1;
      ++result.targeted_voxels;
    } else if (is_reliable_donor_status(status)) {
      donor_mask[index] = 1;
    }
  }

  std::vector<NeighborhoodPlan> plans;
  plans.reserve(result.targeted_voxels);
  const std::size_t plane = height * width;

  for (std::size_t center = 0; center < voxel_count; ++center) {
    if (result.target_mask[center] == 0) {
      continue;
    }

    const std::size_t z = center / plane;
    const std::size_t rem = center % plane;
    const std::size_t y = rem / width;
    const std::size_t x = rem % width;

    const std::size_t z0 = z > static_cast<std::size_t>(options.radius) ? z - static_cast<std::size_t>(options.radius) : 0;
    const std::size_t y0 = y > static_cast<std::size_t>(options.radius) ? y - static_cast<std::size_t>(options.radius) : 0;
    const std::size_t x0 = x > static_cast<std::size_t>(options.radius) ? x - static_cast<std::size_t>(options.radius) : 0;
    const std::size_t z1 = std::min(depth - 1, z + static_cast<std::size_t>(options.radius));
    const std::size_t y1 = std::min(height - 1, y + static_cast<std::size_t>(options.radius));
    const std::size_t x1 = std::min(width - 1, x + static_cast<std::size_t>(options.radius));

    NeighborhoodPlan plan;
    plan.center = center;
    for (std::size_t zz = z0; zz <= z1; ++zz) {
      for (std::size_t yy = y0; yy <= y1; ++yy) {
        for (std::size_t xx = x0; xx <= x1; ++xx) {
          const std::size_t neighbor = flatten_index(zz, yy, xx, height, width);
          if (donor_mask[neighbor] != 0) {
            plan.donors.push_back(neighbor);
          }
        }
      }
    }

    if (static_cast<int>(plan.donors.size()) >= options.min_reliable_neighbors) {
      result.applied_mask[center] = 1;
      ++result.applied_voxels;
      plans.push_back(std::move(plan));
    } else {
      result.unresolved_mask[center] = 1;
    }
  }

  if (plans.empty()) {
    return result;
  }

  const std::vector<double> k1_filtered =
      apply_plan_to_map(k1_map, plans, options.min_reliable_neighbors);
  const std::vector<double> k2_filtered =
      apply_plan_to_map(k2_map, plans, options.min_reliable_neighbors);
  const std::vector<double> k3_filtered =
      apply_plan_to_map(k3_map, plans, options.min_reliable_neighbors);
  const std::vector<double> vb_filtered =
      apply_plan_to_map(vb_map, plans, options.min_reliable_neighbors);
  std::vector<double> delay_filtered;
  if (delay_map != nullptr) {
    if (delay_map->size() != voxel_count) {
      throw std::runtime_error("Delay map size mismatch");
    }
    delay_filtered = apply_plan_to_map(*delay_map, plans, options.min_reliable_neighbors);
  }
  std::vector<double> dispersion_filtered;
  if (dispersion_map != nullptr) {
    if (dispersion_map->size() != voxel_count) {
      throw std::runtime_error("Dispersion map size mismatch");
    }
    dispersion_filtered = apply_plan_to_map(*dispersion_map, plans, options.min_reliable_neighbors);
  }

  std::vector<double> k4_filtered;
  if (model == ModelKind::Reversible2TCM) {
    k4_filtered = apply_plan_to_map(*k4_map, plans, options.min_reliable_neighbors);
  }

  k1_map = k1_filtered;
  k2_map = k2_filtered;
  k3_map = k3_filtered;
  vb_map = vb_filtered;
  if (delay_map != nullptr) {
    *delay_map = delay_filtered;
  }
  if (dispersion_map != nullptr) {
    *dispersion_map = dispersion_filtered;
  }
  if (model == ModelKind::Reversible2TCM) {
    *k4_map = k4_filtered;
  }

  for (const NeighborhoodPlan& plan : plans) {
    ParameterVector params{};
    params.K1 = k1_map[plan.center];
    params.k2 = k2_map[plan.center];
    params.k3 = k3_map[plan.center];
    params.k4 = model == ModelKind::Reversible2TCM ? (*k4_map)[plan.center] : 0.0;
    params.vB = vb_map[plan.center];
    params.delay = delay_map != nullptr ? (*delay_map)[plan.center] : 0.0;
    params.dispersion = dispersion_map != nullptr ? (*dispersion_map)[plan.center] : 0.0;
    ki_map[plan.center] = compute_ki(params);
  }

  return result;
}

}  // namespace tcm
