#pragma once

#include "parameters.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace tcm {

inline double dot(
    const std::array<double, kMaxParameterCount>& a,
    const std::array<double, kMaxParameterCount>& b,
    int n) {
  double value = 0.0;
  for (int i = 0; i < n; ++i) {
    value += a[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
  }
  return value;
}

inline double inf_norm(const std::array<double, kMaxParameterCount>& x, int n) {
  double value = 0.0;
  for (int i = 0; i < n; ++i) {
    value = std::max(value, std::abs(x[static_cast<std::size_t>(i)]));
  }
  return value;
}

inline double l2_norm(const std::array<double, kMaxParameterCount>& x, int n) {
  return std::sqrt(dot(x, x, n));
}

inline double quad_form(
    const std::array<double, kMaxMatrixSize>& matrix,
    const std::array<double, kMaxParameterCount>& x,
    int n) {
  double value = 0.0;
  for (int row = 0; row < n; ++row) {
    double row_dot = 0.0;
    for (int col = 0; col < n; ++col) {
      row_dot += matrix[static_cast<std::size_t>(row * kMaxParameterCount + col)] *
          x[static_cast<std::size_t>(col)];
    }
    value += x[static_cast<std::size_t>(row)] * row_dot;
  }
  return value;
}

inline bool solve_linear_system(
    std::array<double, kMaxMatrixSize> matrix,
    const std::array<double, kMaxParameterCount>& rhs,
    int n,
    std::array<double, kMaxParameterCount>& solution) {
  std::array<double, kMaxParameterCount> b = rhs;

  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double pivot_abs = std::abs(matrix[static_cast<std::size_t>(col * kMaxParameterCount + col)]);
    for (int row = col + 1; row < n; ++row) {
      const double value = std::abs(matrix[static_cast<std::size_t>(row * kMaxParameterCount + col)]);
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
            matrix[static_cast<std::size_t>(col * kMaxParameterCount + k)],
            matrix[static_cast<std::size_t>(pivot * kMaxParameterCount + k)]);
      }
      std::swap(b[static_cast<std::size_t>(col)], b[static_cast<std::size_t>(pivot)]);
    }

    const double diag = matrix[static_cast<std::size_t>(col * kMaxParameterCount + col)];
    for (int row = col + 1; row < n; ++row) {
      const double factor = matrix[static_cast<std::size_t>(row * kMaxParameterCount + col)] / diag;
      if (factor == 0.0) {
        continue;
      }
      for (int k = col; k < n; ++k) {
        matrix[static_cast<std::size_t>(row * kMaxParameterCount + k)] -=
            factor * matrix[static_cast<std::size_t>(col * kMaxParameterCount + k)];
      }
      b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(col)];
    }
  }

  solution.fill(0.0);
  for (int row = n - 1; row >= 0; --row) {
    double value = b[static_cast<std::size_t>(row)];
    for (int col = row + 1; col < n; ++col) {
      value -= matrix[static_cast<std::size_t>(row * kMaxParameterCount + col)] *
          solution[static_cast<std::size_t>(col)];
    }
    const double diag = matrix[static_cast<std::size_t>(row * kMaxParameterCount + row)];
    if (std::abs(diag) < 1e-14) {
      return false;
    }
    solution[static_cast<std::size_t>(row)] = value / diag;
  }

  return true;
}

}  // namespace tcm
