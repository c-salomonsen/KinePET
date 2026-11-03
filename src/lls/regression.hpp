#pragma once

#include "tcm_model.hpp"

namespace tcm::lls {

using Coefficients = std::array<double, kMaxParameterCount>;

// All integrals use minutes and start at the first supplied sample.
// Repeated trapezoidal integration and ridge=1e-6 preserve the legacy warm start.
std::vector<double> cumulative_trapezoid(
    const std::vector<double>& signal, const std::vector<double>& time_minutes);
bool solve_ridge(const std::vector<std::vector<double>>& design,
                 const std::vector<double>& response, Coefficients& coefficients, int count);

struct LinearInputs {
  const std::vector<double>& time;
  const std::vector<double>& aif;
  const std::vector<double>& tac;
  std::vector<double> integral_aif;
  std::vector<double> double_integral_aif;

  LinearInputs(const std::vector<double>& time_minutes,
               const std::vector<double>& input, const std::vector<double>& measured);
};

// q = (m - vB*C_A)/(1 - vB). Preserve the warm-start-only [0, .95] clamp.
std::vector<double> tissue_curve(const LinearInputs& data, double vb);
constexpr double kRecoveryEpsilon = 1e-8;

}  // namespace tcm::lls
