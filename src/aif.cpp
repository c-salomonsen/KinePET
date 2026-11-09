#include "aif.hpp"

#include <algorithm>
#include <cmath>

namespace tcm {
namespace {

struct SampledCurve {
  double value = 0.0;
  double derivative = 0.0;
};

SampledCurve sample_piecewise_linear_with_derivative(
    const std::vector<double>& time_minutes,
    const std::vector<double>& signal,
    double t) {
  SampledCurve sampled{};
  if (time_minutes.empty() || signal.empty()) {
    return sampled;
  }
  if (t < 0.0) {
    return sampled;
  }
  if (time_minutes.size() == 1) {
    if (time_minutes[0] > 0.0 && t < time_minutes[0]) {
      sampled.value = signal[0] * (t / time_minutes[0]);
      sampled.derivative = signal[0] / time_minutes[0];
      return sampled;
    }
    sampled.value = signal[0];
    return sampled;
  }
  if (t < time_minutes[0]) {
    if (time_minutes[0] <= 0.0) {
      sampled.value = signal[0];
      return sampled;
    }
    sampled.value = signal[0] * (t / time_minutes[0]);
    sampled.derivative = signal[0] / time_minutes[0];
    return sampled;
  }
  if (t > time_minutes.back()) {
    sampled.value = signal.back();
    return sampled;
  }

  const auto upper = std::upper_bound(time_minutes.begin(), time_minutes.end(), t);
  const std::size_t idx = static_cast<std::size_t>(std::max<std::ptrdiff_t>(0, upper - time_minutes.begin() - 1));
  if (idx + 1 >= time_minutes.size()) {
    sampled.value = signal.back();
    return sampled;
  }

  const double t0 = time_minutes[idx];
  const double t1 = time_minutes[idx + 1];
  if (t1 <= t0) {
    sampled.value = signal[idx];
    return sampled;
  }
  const double slope = (signal[idx + 1] - signal[idx]) / (t1 - t0);
  sampled.value = signal[idx] + (t - t0) * slope;
  sampled.derivative = slope;
  return sampled;
}

}  // namespace

void transform_aif(
    const std::vector<double>& time_minutes,
    const std::vector<double>& measured_aif,
    double delay,
    double dispersion,
    std::vector<double>& transformed) {
  // Preserve the measured samples when a caller requests an in-place shift.
  if (&measured_aif == &transformed) {
    const auto source = measured_aif;
    transform_aif(time_minutes, source, delay, dispersion, transformed);
    return;
  }
  transformed.resize(time_minutes.size());
  const double tau = std::max(0.0, dispersion);
  for (std::size_t i = 0; i < time_minutes.size(); ++i) {
    const SampledCurve sampled =
        sample_piecewise_linear_with_derivative(time_minutes, measured_aif, time_minutes[i] + delay);
    transformed[i] = std::max(0.0, sampled.value + tau * sampled.derivative);
  }
}

}  // namespace tcm
