// SPDX-FileCopyrightText: 2026 Christian Salomonsen
// SPDX-License-Identifier: AGPL-3.0-only

#include "tissue.hpp"
#include "convolution.hpp"

#include <algorithm>
#include <cmath>

namespace tcm::kinetics {
namespace {

bool evaluate_reversible_fallback(
    const ParameterVector& params,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    std::vector<double>& tissue) {
  const auto cp_at = [&](double t, std::size_t interval) {
    if (interval + 1 >= time_minutes.size()) {
      return aif.back();
    }
    const double t0 = time_minutes[interval];
    const double t1 = time_minutes[interval + 1];
    if (t1 <= t0) {
      return aif[interval];
    }
    const double alpha = (t - t0) / (t1 - t0);
    return aif[interval] + alpha * (aif[interval + 1] - aif[interval]);
  };

  const auto deriv = [&](double c1, double c2, double cp, double& dc1, double& dc2) {
    dc1 = params.K1 * cp - (params.k2 + params.k3) * c1 + params.k4 * c2;
    dc2 = params.k3 * c1 - params.k4 * c2;
  };

  const std::size_t t_count = time_minutes.size();
  tissue.assign(t_count, 0.0);

  double c1 = 0.0;
  double c2 = 0.0;

  auto commit_prediction = [&](std::size_t idx) {
    tissue[idx] = c1 + c2;
  };

  if (time_minutes[0] > 0.0) {
    const double dt0 = time_minutes[0];
    constexpr int substeps = 8;
    const double h = dt0 / static_cast<double>(substeps);
    for (int sub = 0; sub < substeps; ++sub) {
      const double t_local = static_cast<double>(sub) * h;
      const double cp0 = (aif[0] / dt0) * t_local;
      const double cp1 = (aif[0] / dt0) * (t_local + 0.5 * h);
      const double cp2 = (aif[0] / dt0) * (t_local + h);
      double k1_c1 = 0.0;
      double k1_c2 = 0.0;
      double k2_c1 = 0.0;
      double k2_c2 = 0.0;
      double k3_c1 = 0.0;
      double k3_c2 = 0.0;
      double k4_c1 = 0.0;
      double k4_c2 = 0.0;
      deriv(c1, c2, cp0, k1_c1, k1_c2);
      deriv(c1 + 0.5 * h * k1_c1, c2 + 0.5 * h * k1_c2, cp1, k2_c1, k2_c2);
      deriv(c1 + 0.5 * h * k2_c1, c2 + 0.5 * h * k2_c2, cp1, k3_c1, k3_c2);
      deriv(c1 + h * k3_c1, c2 + h * k3_c2, cp2, k4_c1, k4_c2);
      c1 += (h / 6.0) * (k1_c1 + 2.0 * k2_c1 + 2.0 * k3_c1 + k4_c1);
      c2 += (h / 6.0) * (k1_c2 + 2.0 * k2_c2 + 2.0 * k3_c2 + k4_c2);
    }
  }

  commit_prediction(0);

  for (std::size_t idx = 0; idx + 1 < t_count; ++idx) {
    const double dt = time_minutes[idx + 1] - time_minutes[idx];
    if (dt <= 0.0) {
      return false;
    }

    constexpr int substeps = 8;
    const double h = dt / static_cast<double>(substeps);
    for (int sub = 0; sub < substeps; ++sub) {
      const double t_start = time_minutes[idx] + static_cast<double>(sub) * h;
      const double cp0 = cp_at(t_start, idx);
      const double cp1 = cp_at(t_start + 0.5 * h, idx);
      const double cp2 = cp_at(t_start + h, idx);
      double k1_c1 = 0.0;
      double k1_c2 = 0.0;
      double k2_c1 = 0.0;
      double k2_c2 = 0.0;
      double k3_c1 = 0.0;
      double k3_c2 = 0.0;
      double k4_c1 = 0.0;
      double k4_c2 = 0.0;
      deriv(c1, c2, cp0, k1_c1, k1_c2);
      deriv(c1 + 0.5 * h * k1_c1, c2 + 0.5 * h * k1_c2, cp1, k2_c1, k2_c2);
      deriv(c1 + 0.5 * h * k2_c1, c2 + 0.5 * h * k2_c2, cp1, k3_c1, k3_c2);
      deriv(c1 + h * k3_c1, c2 + h * k3_c2, cp2, k4_c1, k4_c2);
      c1 += (h / 6.0) * (k1_c1 + 2.0 * k2_c1 + 2.0 * k3_c1 + k4_c1);
      c2 += (h / 6.0) * (k1_c2 + 2.0 * k2_c2 + 2.0 * k3_c2 + k4_c2);
    }

    commit_prediction(idx + 1);
  }

  return true;
}
}  // namespace

// q'' + (k2+k3+k4)*q' + k2*k4*q = K1*C_A' + K1*(k3+k4)*C_A.
// Its transfer function is K1*(s+k3+k4)/(s*s+(k2+k3+k4)*s+k2*k4).
bool evaluate_tissue(const Reversible2TCM&, const ParameterVector& params,
                     const std::vector<double>& time_minutes,
                     const std::vector<double>& aif, std::vector<double>& tissue) {
  const std::size_t t_count = time_minutes.size();
  tissue.assign(t_count, 0.0);
  const double rate_sum = params.k2 + params.k3 + params.k4;
  const double discriminant =
      std::max(0.0, rate_sum * rate_sum - 4.0 * params.k2 * params.k4);
  const double root = std::sqrt(discriminant);
  const double a1 = 0.5 * (rate_sum + root);
  const double a2 = 0.5 * (rate_sum - root);
  const double denom = a1 - a2;

  if (std::abs(denom) < 1e-8) {
    return evaluate_reversible_fallback(params, time_minutes, aif, tissue);
  }

  const double c1 = params.K1 * (a1 - params.k3 - params.k4) / denom;
  const double c2 = -params.K1 * (a2 - params.k3 - params.k4) / denom;

  double z1 = 0.0;
  double z2 = 0.0;
  if (time_minutes[0] > 0.0) {
    z1 = exp_conv_step(0.0, a1, time_minutes[0], 0.0, aif[0]);
    z2 = exp_conv_step(0.0, a2, time_minutes[0], 0.0, aif[0]);
  }

  tissue[0] = c1 * z1 + c2 * z2;

  for (std::size_t idx = 0; idx + 1 < t_count; ++idx) {
    const double dt = time_minutes[idx + 1] - time_minutes[idx];
    if (dt <= 0.0) {
      return false;
    }

    z1 = exp_conv_step(z1, a1, dt, aif[idx], aif[idx + 1]);
    z2 = exp_conv_step(z2, a2, dt, aif[idx], aif[idx + 1]);
    tissue[idx + 1] = c1 * z1 + c2 * z2;
  }

  return true;
}

}  // namespace tcm::kinetics
