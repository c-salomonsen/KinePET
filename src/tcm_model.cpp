// SPDX-FileCopyrightText: 2026 Christian Salomonsen
// SPDX-License-Identifier: AGPL-3.0-only

#include "tcm_model.hpp"
#include "aif.hpp"
#include "models/tissue.hpp"
#include "validation.hpp"

#include <algorithm>
#include <cmath>

namespace tcm {

bool evaluate_model(
    const Model& model,
    const ParameterVector& raw_params,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif_raw,
    std::vector<double>& predicted) {
  if (time_minutes.size() != aif_raw.size() || !valid_time(time_minutes) ||
      !all_finite(aif_raw) || !finite_parameters(raw_params)) {
    return false;
  }

  ParameterVector params = raw_params;
  params.vB = std::clamp(params.vB, 0.0, 1.0);
  params.dispersion = std::max(0.0, params.dispersion);
  if (params.K1 < 0.0 || params.k2 < 0.0 || params.k3 < 0.0 || params.k4 < 0.0) {
    return false;
  }

  std::vector<double> aif = aif_raw;
  for (double& value : aif) {
    if (value < 1e-5) {
      value = 0.0;
    }
  }
  if (std::abs(params.delay) > 1e-12 || params.dispersion > 1e-12) {
    std::vector<double> transformed;
    transform_aif(time_minutes, aif, params.delay, params.dispersion, transformed);
    aif = std::move(transformed);
    if (!all_finite(aif)) return false;
  }

  const bool ok = std::visit([&](const auto& kinetic_model) {
    return kinetics::evaluate_tissue(kinetic_model, params, time_minutes, aif, predicted);
  }, model);
  if (!ok) return false;

  // m = (1-vB)*q + vB*C_A, for either fitted or fixed blood fraction.
  for (std::size_t i = 0; i < predicted.size(); ++i) {
    predicted[i] = (1.0 - params.vB) * predicted[i] + params.vB * aif[i];
  }
  return all_finite(predicted);
}

}  // namespace tcm
