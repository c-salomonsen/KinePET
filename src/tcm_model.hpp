#pragma once

#include "parameters.hpp"

namespace tcm {

// The single compiled forward-model entrypoint: normalize/transform the AIF,
// dispatch tissue kinetics by model type, then add the vascular contribution.
bool evaluate_model(
    const Model& model,
    const ParameterVector& params,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    std::vector<double>& predicted);

}  // namespace tcm
