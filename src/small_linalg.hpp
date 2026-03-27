#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace tcm {

inline double dot(const std::array<double, 5>& a, const std::array<double, 5>& b, int n) {
  double value = 0.0;
  for (int i = 0; i < n; ++i) {
    value += a[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
  }
  return value;
}

inline double inf_norm(const std::array<double, 5>& x, int n) {
  double value = 0.0;
  for (int i = 0; i < n; ++i) {
    value = std::max(value, std::abs(x[static_cast<std::size_t>(i)]));
  }
  return value;
}

inline double l2_norm(const std::array<double, 5>& x, int n) {
  return std::sqrt(dot(x, x, n));
}

inline double quad_form(
    const std::array<double, 25>& matrix,
    const std::array<double, 5>& x,
    int n) {
  double value = 0.0;
  for (int row = 0; row < n; ++row) {
    double row_dot = 0.0;
    for (int col = 0; col < n; ++col) {
      row_dot += matrix[static_cast<std::size_t>(row * 5 + col)] *
          x[static_cast<std::size_t>(col)];
    }
    value += x[static_cast<std::size_t>(row)] * row_dot;
  }
  return value;
}

inline bool solve_linear_system(
    std::array<double, 25> matrix,
    const std::array<double, 5>& rhs,
    int n,
    std::array<double, 5>& solution) {
  std::array<double, 5> b = rhs;

  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double pivot_abs = std::abs(matrix[static_cast<std::size_t>(col * 5 + col)]);
    for (int row = col + 1; row < n; ++row) {
      const double value = std::abs(matrix[static_cast<std::size_t>(row * 5 + col)]);
      if (value > pivot_abs) {
        pivot = row;
        pivot_abs = value;
      }
    }

    if (pivot_abs < 1e-14) {
      return false;
    }

    if (pivot != col) {
      for (int k = col; k < n; ++k) {
        std::swap(
            matrix[static_cast<std::size_t>(col * 5 + k)],
            matrix[static_cast<std::size_t>(pivot * 5 + k)]);
      }
      std::swap(b[static_cast<std::size_t>(col)], b[static_cast<std::size_t>(pivot)]);
    }

    const double diag = matrix[static_cast<std::size_t>(col * 5 + col)];
    for (int row = col + 1; row < n; ++row) {
      const double factor = matrix[static_cast<std::size_t>(row * 5 + col)] / diag;
      if (factor == 0.0) {
        continue;
      }
      for (int k = col; k < n; ++k) {
        matrix[static_cast<std::size_t>(row * 5 + k)] -=
            factor * matrix[static_cast<std::size_t>(col * 5 + k)];
      }
      b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(col)];
    }
  }

  solution.fill(0.0);
  for (int row = n - 1; row >= 0; --row) {
    double value = b[static_cast<std::size_t>(row)];
    for (int col = row + 1; col < n; ++col) {
      value -= matrix[static_cast<std::size_t>(row * 5 + col)] *
          solution[static_cast<std::size_t>(col)];
    }
    const double diag = matrix[static_cast<std::size_t>(row * 5 + row)];
    if (std::abs(diag) < 1e-14) {
      return false;
    }
    solution[static_cast<std::size_t>(row)] = value / diag;
  }

  return true;
}

}  // namespace tcm
