#include "formulations.hpp"

#include <algorithm>
#include <cmath>

namespace tcm::lls {
namespace {

bool warm_start_reversible_fitted_vb(const LinearInputs& data, ParameterVector& params) {
  const auto integral_tac = cumulative_trapezoid(data.tac, data.time);
  const auto double_integral_tac = cumulative_trapezoid(integral_tac, data.time);
  // m = a1*integral(m) + a2*double_integral(m) + a3*C_A
  //     + a4*integral(C_A) + a5*double_integral(C_A).
  std::vector<std::vector<double>> design(data.tac.size(), std::vector<double>(5, 0.0));
  for (std::size_t i = 0; i < data.tac.size(); ++i) {
    design[i][0] = integral_tac[i];
    design[i][1] = double_integral_tac[i];
    design[i][2] = data.aif[i];
    design[i][3] = data.integral_aif[i];
    design[i][4] = data.double_integral_aif[i];
  }
  Coefficients coefficients{};
  return solve_ridge(design, data.tac, coefficients, 5) &&
      recover_reversible_fitted_vb(coefficients, params);
}

bool warm_start_reversible_fixed_vb(const LinearInputs& data, FixedVB vb, ParameterVector& params) {
  params.vB = std::clamp(vb.value, 0.0, 0.95);
  const auto tissue = tissue_curve(data, params.vB);
  const auto integral_tissue = cumulative_trapezoid(tissue, data.time);
  const auto double_integral_tissue = cumulative_trapezoid(integral_tissue, data.time);
  // q = b1*integral(q) + b2*double_integral(q)
  //     + b3*integral(C_A) + b4*double_integral(C_A).
  std::vector<std::vector<double>> design(tissue.size(), std::vector<double>(4, 0.0));
  for (std::size_t i = 0; i < tissue.size(); ++i) {
    design[i][0] = integral_tissue[i];
    design[i][1] = double_integral_tissue[i];
    design[i][2] = data.integral_aif[i];
    design[i][3] = data.double_integral_aif[i];
  }
  Coefficients coefficients{};
  return solve_ridge(design, tissue, coefficients, 4) &&
      recover_reversible_fixed_vb(coefficients, params);
}

}  // namespace

bool recover_reversible_fitted_vb(const Coefficients& coefficients, ParameterVector& params) {
  // alpha=k2+k3+k4, beta=k2*k4, gamma=K1*(k3+k4).
  // a1=-alpha, a2=-beta, a3=vB,
  // a4=vB*alpha+(1-vB)*K1, a5=vB*beta+(1-vB)*gamma.
  // These a1..a5 follow the derivation's regressor order above exactly.
  const double a1 = coefficients[0];
  const double a2 = coefficients[1];
  const double a3 = coefficients[2];
  const double a4 = coefficients[3];
  const double a5 = coefficients[4];
  const double tissue_fraction = 1.0 - a3;
  if (std::abs(tissue_fraction) <= kRecoveryEpsilon) return false;

  const double d0 = a4 + a1 * a3;  // (1-vB)*K1
  params.K1 = d0 / tissue_fraction;
  if (params.K1 <= kRecoveryEpsilon) return false;
  const double s = (a5 + a3 * a2) / d0;  // k3+k4
  params.k2 = -a1 - s;
  if (params.k2 <= kRecoveryEpsilon) return false;
  params.k4 = -a2 / params.k2;
  params.k3 = s - params.k4;
  params.vB = a3;
  return true;
}

bool recover_reversible_fixed_vb(const Coefficients& coefficients, ParameterVector& params) {
  // b1=-(k2+k3+k4), b2=-k2*k4, b3=K1, b4=K1*(k3+k4).
  params.K1 = coefficients[2];
  if (params.K1 <= kRecoveryEpsilon) return false;
  // Preserve the legacy epsilon in this fixed-vB inversion.
  const double s = coefficients[3] / (params.K1 + kRecoveryEpsilon);
  params.k2 = -coefficients[0] - s;
  if (std::abs(params.k2) <= kRecoveryEpsilon) return false;
  params.k4 = -coefficients[1] / params.k2;
  params.k3 = s - params.k4;
  return true;
}

bool estimate_linear_parameters(const LinearInputs& data, const Reversible2TCM& model,
                                ParameterVector& params) {
  if (std::holds_alternative<FittedVB>(model.vb)) {
    return warm_start_reversible_fitted_vb(data, params);
  }
  return warm_start_reversible_fixed_vb(data, std::get<FixedVB>(model.vb), params);
}

}  // namespace tcm::lls
