#include "solver.hpp"

#include "small_linalg.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace tcm {
namespace {

double stable_sigmoid(double value) {
  if (value >= 0.0) {
    const double z = std::exp(-value);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(value);
  return z / (1.0 + z);
}

double stable_logit(double value) {
  return std::log(value / (1.0 - value));
}

struct Workspace {
  int n_params = 0;
  std::vector<double> predicted;
  std::vector<double> residuals;
  std::vector<double> trial_predicted;
  std::vector<double> trial_residuals;
  std::vector<double> plus_predicted;
  std::vector<double> minus_predicted;
  std::array<double, kMaxParameterCount> u{};
  std::array<double, kMaxParameterCount> trial_u{};
  std::array<double, kMaxParameterCount> gradient{};
  std::array<double, kMaxMatrixSize> hessian{};
  std::array<double, kMaxParameterCount> step{};
  std::vector<double> jacobian;
};

std::array<double, kMaxParameterCount> active_lower(
    const SolverBounds& bounds,
    const Model& model,
    const AifOptions& aif_options) {
  std::array<double, kMaxParameterCount> out{};
  out.fill(0.0);
  const int n = active_parameter_count(model, aif_options);
  for (int i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = bounds.lower[static_cast<std::size_t>(i)];
  }
  return out;
}

std::array<double, kMaxParameterCount> active_upper(
    const SolverBounds& bounds,
    const Model& model,
    const AifOptions& aif_options) {
  std::array<double, kMaxParameterCount> out{};
  out.fill(0.0);
  const int n = active_parameter_count(model, aif_options);
  for (int i = 0; i < n; ++i) {
    out[static_cast<std::size_t>(i)] = bounds.upper[static_cast<std::size_t>(i)];
  }
  return out;
}

void x_to_u(
    const std::array<double, kMaxParameterCount>& x,
    const std::array<double, kMaxParameterCount>& lower,
    const std::array<double, kMaxParameterCount>& upper,
    int n,
    std::array<double, kMaxParameterCount>& u) {
  constexpr double eps = 1e-8;
  for (int i = 0; i < n; ++i) {
    const double span = upper[static_cast<std::size_t>(i)] - lower[static_cast<std::size_t>(i)];
    double y = (x[static_cast<std::size_t>(i)] - lower[static_cast<std::size_t>(i)]) / span;
    y = std::clamp(y, eps, 1.0 - eps);
    u[static_cast<std::size_t>(i)] = stable_logit(y);
  }
}

ParameterVector u_to_params(
    const std::array<double, kMaxParameterCount>& u,
    const SolverOptions& options) {
  const int n = active_parameter_count(
      options.model, options.aif);
  std::array<double, kMaxParameterCount> x{};
  const auto lower = active_lower(
      options.bounds,
      options.model, options.aif);
  const auto upper = active_upper(
      options.bounds,
      options.model, options.aif);
  for (int i = 0; i < n; ++i) {
    const double s = stable_sigmoid(u[static_cast<std::size_t>(i)]);
    x[static_cast<std::size_t>(i)] =
        lower[static_cast<std::size_t>(i)] +
        (upper[static_cast<std::size_t>(i)] - lower[static_cast<std::size_t>(i)]) * s;
  }
  return unpack_active(
      x,
      options.model, options.aif);
}

bool evaluate_weighted_residuals(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& tac,
    const std::array<double, kMaxParameterCount>& u,
    std::vector<double>& predicted,
    std::vector<double>& residuals,
    double& cost,
    double& rmse,
    ParameterVector& params) {
  params = u_to_params(u, options);
  if (!evaluate_model(options.model, params, time_minutes, aif, predicted)) {
    return false;
  }

  cost = 0.0;
  double sq = 0.0;
  residuals.resize(tac.size());
  for (std::size_t i = 0; i < tac.size(); ++i) {
    const double diff = predicted[i] - tac[i];
    residuals[i] = diff * weights[i];
    cost += 0.5 * residuals[i] * residuals[i];
    sq += diff * diff;
  }
  rmse = std::sqrt(sq / static_cast<double>(tac.size()));
  return true;
}

bool compute_jacobian(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& tac,
    Workspace& work) {
  const std::size_t t_count = tac.size();
  work.jacobian.assign(t_count * static_cast<std::size_t>(work.n_params), 0.0);

  for (int col = 0; col < work.n_params; ++col) {
    const double step = options.fd_step * std::max(1.0, std::abs(work.u[static_cast<std::size_t>(col)]));

    work.trial_u = work.u;
    work.trial_u[static_cast<std::size_t>(col)] += step;

    double plus_cost = 0.0;
    double plus_rmse = 0.0;
    ParameterVector plus_params{};
    if (!evaluate_weighted_residuals(
            options,
            time_minutes,
            aif,
            weights,
            tac,
            work.trial_u,
            work.plus_predicted,
            work.trial_residuals,
            plus_cost,
            plus_rmse,
            plus_params)) {
      return false;
    }
    std::vector<double> residual_plus = work.trial_residuals;

    work.trial_u = work.u;
    work.trial_u[static_cast<std::size_t>(col)] -= step;
    double minus_cost = 0.0;
    double minus_rmse = 0.0;
    ParameterVector minus_params{};
    if (!evaluate_weighted_residuals(
            options,
            time_minutes,
            aif,
            weights,
            tac,
            work.trial_u,
            work.minus_predicted,
            work.trial_residuals,
            minus_cost,
            minus_rmse,
            minus_params)) {
      return false;
    }
    std::vector<double> residual_minus = work.trial_residuals;

    const double inv_step = 1.0 / (2.0 * step);
    for (std::size_t row = 0; row < t_count; ++row) {
      work.jacobian[row * static_cast<std::size_t>(work.n_params) + static_cast<std::size_t>(col)] =
          (residual_plus[row] - residual_minus[row]) * inv_step;
    }
  }

  return true;
}

void build_normal_equations(Workspace& work) {
  work.gradient.fill(0.0);
  work.hessian.fill(0.0);

  const std::size_t t_count = work.residuals.size();
  for (std::size_t row = 0; row < t_count; ++row) {
    for (int i = 0; i < work.n_params; ++i) {
      const double ji = work.jacobian[row * static_cast<std::size_t>(work.n_params) + static_cast<std::size_t>(i)];
      work.gradient[static_cast<std::size_t>(i)] += ji * work.residuals[row];
      for (int j = 0; j < work.n_params; ++j) {
        const double jj =
            work.jacobian[row * static_cast<std::size_t>(work.n_params) + static_cast<std::size_t>(j)];
        work.hessian[static_cast<std::size_t>(i * kMaxParameterCount + j)] += ji * jj;
      }
    }
  }
}

bool solve_lm_step(Workspace& work, double lambda) {
  std::array<double, kMaxMatrixSize> system = work.hessian;
  std::array<double, kMaxParameterCount> rhs{};
  rhs.fill(0.0);

  for (int i = 0; i < work.n_params; ++i) {
    system[static_cast<std::size_t>(i * kMaxParameterCount + i)] +=
        lambda * std::max(1e-12, work.hessian[static_cast<std::size_t>(i * kMaxParameterCount + i)]);
    rhs[static_cast<std::size_t>(i)] = -work.gradient[static_cast<std::size_t>(i)];
  }

  return solve_linear_system(system, rhs, work.n_params, work.step);
}

FitResult initial_result(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& tac,
    Workspace& work,
    const ParameterVector& start_params) {
  FitResult result;
  result.params = start_params;
  result.ki = compute_ki(start_params);
  result.status = FitStatus::InvalidInput;

  std::array<double, kMaxParameterCount> x{};
  pack_active(
      start_params,
      options.model, options.aif,
      x);
  const auto lower = active_lower(
      options.bounds,
      options.model, options.aif);
  const auto upper = active_upper(
      options.bounds,
      options.model, options.aif);
  x_to_u(x, lower, upper, work.n_params, work.u);

  double cost = 0.0;
  double rmse = 0.0;
  ParameterVector params{};
  if (!evaluate_weighted_residuals(
          options,
          time_minutes,
          aif,
          weights,
          tac,
          work.u,
          work.predicted,
          work.residuals,
          cost,
          rmse,
          params)) {
    return result;
  }

  result.params = params;
  result.ki = compute_ki(params);
  result.weighted_cost = cost;
  result.rmse = rmse;
  result.status = FitStatus::Failed;
  return result;
}

// Initialization is independent of the shared nonlinear optimizer below.
ParameterVector choose_initial_parameters(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& tac,
    const ParameterVector* initial_guess) {
  ParameterVector start = default_initial_guess(
      options.model, options.aif);
  if (initial_guess != nullptr) {
    start = *initial_guess;
  } else if (options.use_linear_warm_start) {
    const WarmStartResult warm = warm_start(
        options.model, options.aif,
        options.bounds,
        time_minutes,
        aif,
        tac);
    if (warm.valid) {
      start = warm.params;
    }
  }
  start = clip_to_bounds(
      start,
      options.model, options.aif,
      options.bounds);

  return start;
}

FitResult fit_tac_impl(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& tac,
    const ParameterVector* initial_guess) {
  FitResult result;
  if (time_minutes.size() != tac.size() || aif.size() != tac.size() || weights.size() != tac.size() ||
      tac.size() < 4) {
    result.status = FitStatus::InvalidInput;
    return result;
  }

  Workspace work;
  work.n_params = active_parameter_count(
      options.model, options.aif);
  work.predicted.reserve(tac.size());
  work.residuals.reserve(tac.size());
  work.trial_predicted.reserve(tac.size());
  work.trial_residuals.reserve(tac.size());
  work.plus_predicted.reserve(tac.size());
  work.minus_predicted.reserve(tac.size());

  const ParameterVector start = choose_initial_parameters(
      options, time_minutes, aif, tac, initial_guess);

  result = initial_result(options, time_minutes, aif, weights, tac, work, start);
  if (result.status == FitStatus::InvalidInput) {
    return result;
  }

  double current_cost = result.weighted_cost;
  double current_rmse = result.rmse;
  double lambda = options.lambda_initial;
  double nu = 2.0;

  for (int iter = 0; iter < options.max_iterations; ++iter) {
    result.iterations = iter + 1;

    if (!compute_jacobian(options, time_minutes, aif, weights, tac, work)) {
      result.status = FitStatus::Failed;
      return result;
    }
    build_normal_equations(work);

    if (inf_norm(work.gradient, work.n_params) < options.gradient_tolerance) {
      result.status = FitStatus::GradientConverged;
      result.params = u_to_params(work.u, options);
      result.ki = compute_ki(result.params);
      result.weighted_cost = current_cost;
      result.rmse = current_rmse;
      return result;
    }

    bool accepted = false;
    for (int attempt = 0; attempt < 12; ++attempt) {
      if (!solve_lm_step(work, lambda)) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }

      if (l2_norm(work.step, work.n_params) < options.step_tolerance) {
        result.status = FitStatus::StepConverged;
        result.params = u_to_params(work.u, options);
        result.ki = compute_ki(result.params);
        result.weighted_cost = current_cost;
        result.rmse = current_rmse;
        return result;
      }

      work.trial_u = work.u;
      for (int i = 0; i < work.n_params; ++i) {
        work.trial_u[static_cast<std::size_t>(i)] += work.step[static_cast<std::size_t>(i)];
      }

      double trial_cost = 0.0;
      double trial_rmse = 0.0;
      ParameterVector trial_params{};
      if (!evaluate_weighted_residuals(
              options,
              time_minutes,
              aif,
              weights,
              tac,
              work.trial_u,
              work.trial_predicted,
              work.trial_residuals,
              trial_cost,
              trial_rmse,
              trial_params)) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }

      const double predicted_reduction =
          -dot(work.gradient, work.step, work.n_params) -
          0.5 * quad_form(work.hessian, work.step, work.n_params);
      if (predicted_reduction <= 0.0) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }

      const double actual_reduction = current_cost - trial_cost;
      const double rho = actual_reduction / predicted_reduction;

      if (rho > 0.0 && std::isfinite(trial_cost)) {
        work.u = work.trial_u;
        work.predicted = work.trial_predicted;
        work.residuals = work.trial_residuals;
        result.params = trial_params;
        result.ki = compute_ki(trial_params);
        result.weighted_cost = trial_cost;
        result.rmse = trial_rmse;
        current_cost = trial_cost;
        current_rmse = trial_rmse;
        lambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3.0));
        nu = 2.0;
        accepted = true;
        break;
      }

      lambda *= nu;
      nu *= 2.0;
    }

    if (!accepted) {
      result.status = FitStatus::Failed;
      return result;
    }

    if (result.weighted_cost <= options.cost_tolerance) {
      result.status = FitStatus::CostConverged;
      return result;
    }
  }

  result.status = FitStatus::MaxIterations;
  return result;
}

}  // namespace

FitResult fit_tac(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& tac) {
  return fit_tac_impl(options, time_minutes, aif, weights, tac, nullptr);
}

FitResult fit_tac_with_initial_guess(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& tac,
    const ParameterVector& initial_guess) {
  return fit_tac_impl(options, time_minutes, aif, weights, tac, &initial_guess);
}

}  // namespace tcm
