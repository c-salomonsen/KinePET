#pragma once

#include "tcm_model.hpp"

#include <vector>

namespace tcm {

struct WarmStartResult {
  ParameterVector params{};
  bool valid = false;
};

WarmStartResult linear_warm_start(
    const Model& model,
    const AifOptions& aif_options,
    const SolverBounds& bounds,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& tac);

}  // namespace tcm
