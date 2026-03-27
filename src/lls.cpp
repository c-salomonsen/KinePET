#include "lls.hpp"

#include "small_linalg.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace tcm {
namespace {

std::vector<double> cumulative_trapezoid(
    const std::vector<double>& signal,
    const std::vector<double>& time_minutes) {
  const std::size_t n = signal.size();
  std::vector<double> integral(n, 0.0);
  for (std::size_t i = 1; i < n; ++i) {
    const double dt = time_minutes[i] - time_minutes[i - 1];
    integral[i] = integral[i - 1] + 0.5 * (signal[i] + signal[i - 1]) * dt;
  }
  return integral;
}

bool solve_ridge(
    const std::vector<std::vector<double>>& design,
    const std::vector<double>& y,
    double ridge,
    std::array<double, 5>& coeffs,
    int p) {
  std::array<double, 25> gram{};
  std::array<double, 5> rhs{};
  gram.fill(0.0);
  rhs.fill(0.0);

  for (std::size_t row = 0; row < y.size(); ++row) {
    for (int i = 0; i < p; ++i) {
      rhs[static_cast<std::size_t>(i)] += design[row][static_cast<std::size_t>(i)] * y[row];
      for (int j = 0; j < p; ++j) {
        gram[static_cast<std::size_t>(i * 5 + j)] +=
            design[row][static_cast<std::size_t>(i)] * design[row][static_cast<std::size_t>(j)];
      }
    }
  }

  for (int i = 0; i < p; ++i) {
    gram[static_cast<std::size_t>(i * 5 + i)] += ridge;
  }

  return solve_linear_system(gram, rhs, p, coeffs);
}

bool finite_positive(double value) {
  return std::isfinite(value) && value >= 0.0;
}

}  // namespace

WarmStartResult linear_warm_start(
    ModelKind model,
    bool fit_vb,
    double fixed_vb,
    const SolverBounds& bounds,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    const std::vector<double>& tac) {
  WarmStartResult result;
  if (time_minutes.size() != aif.size() || tac.size() != aif.size() || tac.size() < 4) {
    return result;
  }

  constexpr double ridge = 1e-6;
  constexpr double eps = 1e-8;

  const std::vector<double> i_cp = cumulative_trapezoid(aif, time_minutes);
  const std::vector<double> i2_cp = cumulative_trapezoid(i_cp, time_minutes);

  std::array<double, 5> coeffs{};
  coeffs.fill(0.0);
  ParameterVector params = default_initial_guess(model, fit_vb, fixed_vb);

  if (fit_vb) {
    if (model == ModelKind::Irreversible2TCM) {
      const std::vector<double> i_tac = cumulative_trapezoid(tac, time_minutes);
      std::vector<std::vector<double>> x(tac.size(), std::vector<double>(4, 0.0));
      for (std::size_t i = 0; i < tac.size(); ++i) {
        x[i][0] = i_tac[i];
        x[i][1] = aif[i];
        x[i][2] = i_cp[i];
        x[i][3] = i2_cp[i];
      }
      if (!solve_ridge(x, tac, ridge, coeffs, 4)) {
        return result;
      }

      const double p1 = coeffs[0];
      const double p2 = coeffs[1];
      const double p3 = coeffs[2];
      const double p4 = coeffs[3];
      const double denom = 1.0 - p2;
      if (std::abs(denom) <= eps) {
        return result;
      }
      params.K1 = (p3 + p1 * p2) / denom;
      if (params.K1 <= eps) {
        return result;
      }
      params.k3 = p4 / (denom * params.K1 + eps);
      params.k2 = -params.k3 - p1;
      params.vB = p2;
    } else {
      const std::vector<double> i_tac = cumulative_trapezoid(tac, time_minutes);
      const std::vector<double> i2_tac = cumulative_trapezoid(i_tac, time_minutes);
      std::vector<std::vector<double>> x(tac.size(), std::vector<double>(5, 0.0));
      for (std::size_t i = 0; i < tac.size(); ++i) {
        x[i][0] = i_tac[i];
        x[i][1] = i2_tac[i];
        x[i][2] = aif[i];
        x[i][3] = i_cp[i];
        x[i][4] = i2_cp[i];
      }
      if (!solve_ridge(x, tac, ridge, coeffs, 5)) {
        return result;
      }

      const double p1 = coeffs[0];
      const double p2 = coeffs[1];
      const double p3 = coeffs[2];
      const double p4 = coeffs[3];
      const double p5 = coeffs[4];
      const double denom = 1.0 - p3;
      if (std::abs(denom) <= eps) {
        return result;
      }

      params.K1 = (p4 + p1 * p3) / denom;
      if (params.K1 <= eps) {
        return result;
      }
      params.k3 = p5 / (p4 + p1 * p3 + eps) + p3 * p2;
      const double b = p1 + params.k3;
      const double c = -p2;
      const double disc = b * b - 4.0 * c;
      if (disc < 0.0) {
        return result;
      }
      const double sqrt_disc = std::sqrt(disc);
      const double root1 = (-b + sqrt_disc) / 2.0;
      const double root2 = (-b - sqrt_disc) / 2.0;
      params.k2 = std::max(root1, root2);
      params.k4 = std::min(root1, root2);
      params.vB = p3;
    }
  } else {
    const double vB = std::clamp(fixed_vb, 0.0, 0.95);
    const double denom = std::max(1.0 - vB, eps);
    std::vector<double> ct(tac.size(), 0.0);
    for (std::size_t i = 0; i < tac.size(); ++i) {
      ct[i] = (tac[i] - vB * aif[i]) / denom;
    }

    if (model == ModelKind::Irreversible2TCM) {
      const std::vector<double> i_ct = cumulative_trapezoid(ct, time_minutes);
      std::vector<std::vector<double>> x(tac.size(), std::vector<double>(3, 0.0));
      for (std::size_t i = 0; i < tac.size(); ++i) {
        x[i][0] = i_ct[i];
        x[i][1] = i_cp[i];
        x[i][2] = i2_cp[i];
      }
      if (!solve_ridge(x, ct, ridge, coeffs, 3)) {
        return result;
      }
      params.K1 = coeffs[1];
      if (params.K1 <= eps) {
        return result;
      }
      params.k3 = coeffs[2] / (params.K1 + eps);
      params.k2 = -params.k3 - coeffs[0];
      params.vB = vB;
    } else {
      const std::vector<double> i_ct = cumulative_trapezoid(ct, time_minutes);
      const std::vector<double> i2_ct = cumulative_trapezoid(i_ct, time_minutes);
      std::vector<std::vector<double>> x(tac.size(), std::vector<double>(4, 0.0));
      for (std::size_t i = 0; i < tac.size(); ++i) {
        x[i][0] = i_ct[i];
        x[i][1] = i2_ct[i];
        x[i][2] = i_cp[i];
        x[i][3] = i2_cp[i];
      }
      if (!solve_ridge(x, ct, ridge, coeffs, 4)) {
        return result;
      }
      params.K1 = coeffs[2];
      if (params.K1 <= eps) {
        return result;
      }
      const double s = coeffs[3] / (params.K1 + eps);
      params.k2 = -coeffs[0] - s;
      if (std::abs(params.k2) <= eps) {
        return result;
      }
      params.k4 = -coeffs[1] / params.k2;
      params.k3 = s - params.k4;
      params.vB = vB;
    }
  }

  if (!finite_positive(params.K1) || !finite_positive(params.k2) || !finite_positive(params.k3) ||
      !std::isfinite(params.vB) || (model == ModelKind::Reversible2TCM && !finite_positive(params.k4))) {
    return result;
  }

  params = clip_to_bounds(params, model, fit_vb, fixed_vb, bounds);
  result.params = params;
  result.valid = true;
  return result;
}

}  // namespace tcm
