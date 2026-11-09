// SPDX-FileCopyrightText: 2026 Christian Salomonsen
// SPDX-License-Identifier: AGPL-3.0-only

#include "formulations.hpp"

#include <algorithm>
#include <cmath>

namespace tcm::lls {
namespace {

bool warm_start_irreversible_fitted_vb(const LinearInputs& data, ParameterVector& params) {
  const auto integral_tac = cumulative_trapezoid(data.tac, data.time);
  // m = a1*integral(m) + a2*C_A + a3*integral(C_A) + a4*double_integral(C_A).
  std::vector<std::vector<double>> design(data.tac.size(), std::vector<double>(4, 0.0));
  for (std::size_t i = 0; i < data.tac.size(); ++i) {
    design[i][0] = integral_tac[i];
    design[i][1] = data.aif[i];
    design[i][2] = data.integral_aif[i];
    design[i][3] = data.double_integral_aif[i];
  }
  Coefficients coefficients{};
  return solve_ridge(design, data.tac, coefficients, 4) &&
      recover_irreversible_fitted_vb(coefficients, params);
}

bool warm_start_irreversible_fixed_vb(const LinearInputs& data, FixedVB vb, ParameterVector& params) {
  params.vB = std::clamp(vb.value, 0.0, 0.95);
  const auto tissue = tissue_curve(data, params.vB);
  const auto integral_tissue = cumulative_trapezoid(tissue, data.time);
  // q = b1*integral(q) + b2*integral(C_A) + b3*double_integral(C_A).
  std::vector<std::vector<double>> design(tissue.size(), std::vector<double>(3, 0.0));
  for (std::size_t i = 0; i < tissue.size(); ++i) {
    design[i][0] = integral_tissue[i];
    design[i][1] = data.integral_aif[i];
    design[i][2] = data.double_integral_aif[i];
  }
  Coefficients coefficients{};
  return solve_ridge(design, tissue, coefficients, 3) &&
      recover_irreversible_fixed_vb(coefficients, params);
}

}  // namespace

bool recover_irreversible_fitted_vb(const Coefficients& coefficients, ParameterVector& params) {
  // a1=-(k2+k3), a2=vB, a3=vB*(k2+k3)+(1-vB)*K1, a4=(1-vB)*K1*k3.
  const double a1 = coefficients[0];
  const double a2 = coefficients[1];
  const double a3 = coefficients[2];
  const double a4 = coefficients[3];
  const double tissue_fraction = 1.0 - a2;
  if (std::abs(tissue_fraction) <= kRecoveryEpsilon) return false;
  params.K1 = (a3 + a1 * a2) / tissue_fraction;
  if (params.K1 <= kRecoveryEpsilon) return false;
  params.k3 = a4 / (tissue_fraction * params.K1 + kRecoveryEpsilon);
  params.k2 = -params.k3 - a1;
  params.vB = a2;
  return true;
}

bool recover_irreversible_fixed_vb(const Coefficients& coefficients, ParameterVector& params) {
  // b1=-(k2+k3), b2=K1, b3=K1*k3.
  params.K1 = coefficients[1];
  if (params.K1 <= kRecoveryEpsilon) return false;
  params.k3 = coefficients[2] / (params.K1 + kRecoveryEpsilon);
  params.k2 = -params.k3 - coefficients[0];
  return true;
}

bool estimate_linear_parameters(const LinearInputs& data, const Irreversible2TCM& model,
                                ParameterVector& params) {
  if (std::holds_alternative<FittedVB>(model.vb)) {
    return warm_start_irreversible_fitted_vb(data, params);
  }
  return warm_start_irreversible_fixed_vb(data, std::get<FixedVB>(model.vb), params);
}

}  // namespace tcm::lls
