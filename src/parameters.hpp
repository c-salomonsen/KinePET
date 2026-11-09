// SPDX-FileCopyrightText: 2026 Christian Salomonsen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "model.hpp"

#include <array>
#include <string>
#include <vector>

namespace tcm {

constexpr int kMaxParameterCount = 7;
constexpr int kMaxMatrixSize = kMaxParameterCount * kMaxParameterCount;

struct ParameterVector {
  double K1 = 0.1;
  double k2 = 0.1;
  double k3 = 0.1;
  double k4 = 0.05;
  double vB = 0.05;
  double delay = 0.0;
  double dispersion = 0.0;
};

struct SolverBounds {
  std::array<double, kMaxParameterCount> lower{};
  std::array<double, kMaxParameterCount> upper{};
};

int active_parameter_count(const Model& model, const AifOptions& aif_options);
std::string parameter_name(
    const Model& model,
    const AifOptions& aif_options,
    int index);
SolverBounds default_bounds(
    const Model& model,
    const AifOptions& aif_options);
ParameterVector default_initial_guess(
    const Model& model,
    const AifOptions& aif_options);
ParameterVector clip_to_bounds(
    const ParameterVector& params,
    const Model& model,
    const AifOptions& aif_options,
    const SolverBounds& bounds);
void pack_active(
    const ParameterVector& params,
    const Model& model,
    const AifOptions& aif_options,
    std::array<double, kMaxParameterCount>& values);
ParameterVector unpack_active(
    const std::array<double, kMaxParameterCount>& values,
    const Model& model,
    const AifOptions& aif_options);
double compute_ki(const ParameterVector& params);

}  // namespace tcm
