// SPDX-FileCopyrightText: 2026 Christian Salomonsen
// SPDX-License-Identifier: AGPL-3.0-only

#include "lls.hpp"
#include "lls/formulations.hpp"

#include <cmath>

namespace tcm {
namespace {

bool finite_nonnegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

bool admissible_kinetics(const ParameterVector& params, const Model& model) {
  return finite_nonnegative(params.K1) && finite_nonnegative(params.k2) &&
      finite_nonnegative(params.k3) && std::isfinite(params.vB) &&
      (!is_reversible(model) || finite_nonnegative(params.k4));
}

}  // namespace

WarmStartResult warm_start(
    const Model& model,
    const AifOptions& aif_options,
    const SolverBounds& bounds,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& tac) {
  if (time_minutes.size() != aif.size() || tac.size() != aif.size() || tac.size() < 4) {
    return {};
  }

  const lls::LinearInputs data(time_minutes, aif, tac);
  ParameterVector params = default_initial_guess(model, aif_options);
  const bool recovered = std::visit([&](const auto& kinetics) {
    return lls::estimate_linear_parameters(data, kinetics, params);
  }, model);
  if (!recovered || !admissible_kinetics(params, model)) return {};

  return {clip_to_bounds(params, model, aif_options, bounds), true};
}

}  // namespace tcm
