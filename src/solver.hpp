#pragma once

#include "lls.hpp"

#include <vector>
#include <limits>

namespace tcm {

enum class FitStatus {
  Failed = 0,
  GradientConverged = 1,
  StepConverged = 2,
  CostConverged = 3,
  MaxIterations = 4,
  InvalidInput = 5,
};

struct SolverOptions {
  Model model = Reversible2TCM{};
  AifOptions aif;
  SolverBounds bounds{};
  int max_iterations = 40;
  double gradient_tolerance = 1e-6;
  double step_tolerance = 1e-6;
  double cost_tolerance = 1e-9;
  double fd_step = 1e-4;
  double lambda_initial = 1e-2;
  bool use_linear_warm_start = true;
};

struct FitResult {
  ParameterVector params{};
  double ki = 0.0;
  double rmse = std::numeric_limits<double>::quiet_NaN();
  double weighted_cost = std::numeric_limits<double>::quiet_NaN();
  int iterations = 0;
  FitStatus status = FitStatus::Failed;
};

bool valid_solver_options(const SolverOptions& options);

FitResult fit_tac(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& tac);

FitResult fit_tac_with_initial_guess(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& tac,
    const ParameterVector& initial_guess);

}  // namespace tcm
