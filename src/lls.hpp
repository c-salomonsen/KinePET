#pragma once

#include "tcm_model.hpp"

#include <vector>

namespace tcm {

struct WarmStartResult {
  ParameterVector params{};
  bool valid = false;
};

WarmStartResult linear_warm_start(
    ModelKind model,
    bool fit_vb,
    double fixed_vb,
    bool fit_delay,
    double fixed_delay,
    bool fit_dispersion,
    double fixed_dispersion,
    const SolverBounds& bounds,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& tac);

}  // namespace tcm
