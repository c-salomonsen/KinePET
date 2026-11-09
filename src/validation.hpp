#pragma once

#include "parameters.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tcm {

template <class Values>
bool all_finite(const Values& values) {
  return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}

inline bool valid_time(const std::vector<double>& time) {
  if (time.empty() || !all_finite(time) || time.front() < 0.0) return false;
  for (std::size_t i = 1; i < time.size(); ++i) {
    if (time[i] <= time[i - 1]) return false;
  }
  return true;
}

inline bool finite_parameters(const ParameterVector& p) {
  return all_finite(std::array{p.K1, p.k2, p.k3, p.k4, p.vB, p.delay, p.dispersion});
}

}  // namespace tcm
