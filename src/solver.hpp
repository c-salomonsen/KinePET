#pragma once

#include "lls.hpp"

#include <vector>

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
  ModelKind model = ModelKind::Reversible2TCM;
  bool fit_vb = true;
  double fixed_vb = 0.05;
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
  double rmse = 0.0;
  double weighted_cost = 0.0;
  int iterations = 0;
  FitStatus status = FitStatus::Failed;
};

FitResult fit_voxel(
    const SolverOptions& options,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& weights,
    const std::vector<double>& tac);

}  // namespace tcm
