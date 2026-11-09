// SPDX-FileCopyrightText: 2026 Christian Salomonsen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "parameters.hpp"

#include <vector>

namespace tcm {

struct WarmStartResult {
  ParameterVector params{};
  bool valid = false;
};

WarmStartResult warm_start(
    const Model& model,
    const AifOptions& aif_options,
    const SolverBounds& bounds,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& tac);

}  // namespace tcm
