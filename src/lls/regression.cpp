#include "regression.hpp"
#include "small_linalg.hpp"

#include <algorithm>

namespace tcm::lls {

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
    std::array<double, kMaxParameterCount>& coeffs,
    int p) {
  constexpr double ridge = 1e-6;
  std::array<double, kMaxMatrixSize> gram{};
  std::array<double, kMaxParameterCount> rhs{};
  gram.fill(0.0);
  rhs.fill(0.0);

  for (std::size_t row = 0; row < y.size(); ++row) {
    for (int i = 0; i < p; ++i) {
      rhs[static_cast<std::size_t>(i)] += design[row][static_cast<std::size_t>(i)] * y[row];
      for (int j = 0; j < p; ++j) {
        gram[static_cast<std::size_t>(i * kMaxParameterCount + j)] +=
            design[row][static_cast<std::size_t>(i)] * design[row][static_cast<std::size_t>(j)];
      }
    }
  }

  for (int i = 0; i < p; ++i) {
    gram[static_cast<std::size_t>(i * kMaxParameterCount + i)] += ridge;
  }

  return solve_linear_system(gram, rhs, p, coeffs);
}

LinearInputs::LinearInputs(const std::vector<double>& time_minutes,
                           const std::vector<double>& input,
                           const std::vector<double>& measured)
    : time(time_minutes), aif(input), tac(measured),
      integral_aif(cumulative_trapezoid(aif, time)),
      double_integral_aif(cumulative_trapezoid(integral_aif, time)) {}

std::vector<double> tissue_curve(const LinearInputs& data, double vb) {
  const double denom = std::max(1.0 - vb, kRecoveryEpsilon);
  std::vector<double> tissue(data.tac.size(), 0.0);
  for (std::size_t i = 0; i < tissue.size(); ++i) {
    tissue[i] = (data.tac[i] - vb * data.aif[i]) / denom;
  }
  return tissue;
}

}  // namespace tcm::lls
