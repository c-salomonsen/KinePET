#include "parameters.hpp"

#include <algorithm>
#include <stdexcept>

namespace tcm {
namespace {
constexpr double kTinyRate = 1e-12;

int base_parameter_count(const Model& model) {
  if (!is_reversible(model)) {
    return fits_vb(model) ? 4 : 3;
  }
  return fits_vb(model) ? 5 : 4;
}

std::size_t append_if_active(
    bool enabled,
    double value,
    std::array<double, kMaxParameterCount>& values,
    std::size_t index) {
  if (enabled) {
    values[index] = value;
    return index + 1;
  }
  return index;
}

double clamp01(double value) {
  return std::clamp(value, 0.0, 1.0);
}

}  // namespace

int active_parameter_count(const Model& model, const AifOptions& aif_options) {
  return base_parameter_count(model) + (aif_options.delay.fitted ? 1 : 0) + (aif_options.dispersion.fitted ? 1 : 0);
}

std::string parameter_name(
    const Model& model,
    const AifOptions& aif_options,
    int index) {
  static constexpr const char* kIrrevFixedVb[] = {"K1", "k2", "k3"};
  static constexpr const char* kIrrevVb[] = {"K1", "k2", "k3", "vB"};
  static constexpr const char* kRevFixedVb[] = {"K1", "k2", "k3", "k4"};
  static constexpr const char* kRevVb[] = {"K1", "k2", "k3", "k4", "vB"};

  const int n = active_parameter_count(model, aif_options);
  if (index < 0 || index >= n) {
    throw std::out_of_range("Invalid parameter index");
  }

  const int base_count = base_parameter_count(model);
  if (index < base_count) {
    if (!is_reversible(model)) {
      return fits_vb(model) ? kIrrevVb[index] : kIrrevFixedVb[index];
    }
    return fits_vb(model) ? kRevVb[index] : kRevFixedVb[index];
  }
  int extra_index = index - base_count;
  if (aif_options.delay.fitted) {
    if (extra_index == 0) {
      return "delay";
    }
    --extra_index;
  }
  if (aif_options.dispersion.fitted && extra_index == 0) {
    return "dispersion";
  }
  throw std::out_of_range("Invalid parameter index");
}

SolverBounds default_bounds(
    const Model& model,
    const AifOptions& aif_options) {
  SolverBounds bounds{};
  bounds.lower.fill(0.0);
  bounds.upper.fill(0.0);

  std::size_t index = 0;
  bounds.lower[index] = 0.0;
  bounds.upper[index++] = 10.0;  // K1
  bounds.lower[index] = 0.0;
  bounds.upper[index++] = 10.0;  // k2
  bounds.lower[index] = 0.0;
  bounds.upper[index++] = 5.0;  // k3
  if (is_reversible(model)) {
    bounds.lower[index] = 0.0;
    bounds.upper[index++] = 1.0;  // k4
  }
  if (fits_vb(model)) {
    bounds.lower[index] = 0.0;
    bounds.upper[index++] = 1.0;
  }
  if (aif_options.delay.fitted) {
    bounds.lower[index] = -0.2;
    bounds.upper[index++] = 0.2;
  }
  if (aif_options.dispersion.fitted) {
    bounds.lower[index] = 0.0;
    bounds.upper[index++] = 0.1;
  }
  return bounds;
}

ParameterVector default_initial_guess(
    const Model& model,
    const AifOptions& aif_options) {
  ParameterVector params{};
  params.K1 = 0.1;
  params.k2 = 0.1;
  params.k3 = 0.1;
  params.k4 = is_reversible(model) ? 0.05 : 0.0;
  params.vB = fits_vb(model) ? 0.05 : fixed_vb(model);
  params.delay = aif_options.delay.fitted ? 0.0 : aif_options.delay.fixed_value;
  params.dispersion = aif_options.dispersion.fitted ? 0.02 : aif_options.dispersion.fixed_value;
  return params;
}

ParameterVector clip_to_bounds(
    const ParameterVector& params,
    const Model& model,
    const AifOptions& aif_options,
    const SolverBounds& bounds) {
  std::array<double, kMaxParameterCount> values{};
  pack_active(params, model, aif_options, values);

  const int n = active_parameter_count(model, aif_options);
  for (int i = 0; i < n; ++i) {
    values[static_cast<std::size_t>(i)] = std::clamp(
        values[static_cast<std::size_t>(i)],
        bounds.lower[static_cast<std::size_t>(i)],
        bounds.upper[static_cast<std::size_t>(i)]);
  }

  return unpack_active(values, model, aif_options);
}

void pack_active(
    const ParameterVector& params,
    const Model& model,
    const AifOptions& aif_options,
    std::array<double, kMaxParameterCount>& values) {
  values.fill(0.0);
  std::size_t index = 0;
  values[index++] = params.K1;
  values[index++] = params.k2;
  values[index++] = params.k3;
  if (is_reversible(model)) values[index++] = params.k4;
  if (fits_vb(model)) values[index++] = params.vB;
  index = append_if_active(aif_options.delay.fitted, params.delay, values, index);
  (void)append_if_active(aif_options.dispersion.fitted, params.dispersion, values, index);
}

ParameterVector unpack_active(
    const std::array<double, kMaxParameterCount>& values,
    const Model& model,
    const AifOptions& aif_options) {
  ParameterVector params{};
  std::size_t index = 0;
  params.K1 = values[index++];
  params.k2 = values[index++];
  params.k3 = values[index++];
  params.k4 = is_reversible(model) ? values[index++] : 0.0;
  params.vB = fits_vb(model) ? values[index++] : fixed_vb(model);
  params.delay = aif_options.delay.fitted ? values[index++] : aif_options.delay.fixed_value;
  params.dispersion = aif_options.dispersion.fitted ? values[index++] : aif_options.dispersion.fixed_value;
  params.vB = clamp01(params.vB);
  params.dispersion = std::max(0.0, params.dispersion);
  return params;
}

double compute_ki(const ParameterVector& params) {
  const double denom = params.k2 + params.k3;
  if (denom <= kTinyRate) {
    return 0.0;
  }
  return params.K1 * params.k3 / denom;
}

}  // namespace tcm
