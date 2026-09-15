#include "tcm_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tcm {
namespace {

constexpr double kTinyRate = 1e-12;

// Below this rate the closed form of exp_conv_step loses all significant
// digits: its i1 term is (1 - decay * (1 + rate * dt)) / rate^2, where the
// numerator vanishes like rate^2, so the ratio cancels catastrophically. The
// reversible model is exposed because its slow eigenvalue is a product ratio,
// a2 = 2*k2*k4 / (s + sqrt(s^2 - 4*k2*k4)), which vanishes whenever k2 or k4
// is small; the irreversible model's rate is the sum k2 + k3 and is immune.
// At 1e-4 the relative error is already ~5e-8 and climbs to ~1e-2 by 1e-7.
constexpr double kSmallRate = 1e-3;

int base_parameter_count(ModelKind model, bool fit_vb) {
  if (model == ModelKind::Irreversible2TCM) {
    return fit_vb ? 4 : 3;
  }
  return fit_vb ? 5 : 4;
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

void transform_aif(
    const std::vector<double>& time_minutes,
    const std::vector<double>& measured_aif,
    double delay,
    double dispersion,
    std::vector<double>& transformed) {
  transformed.resize(time_minutes.size());
  const double tau = std::max(0.0, dispersion);
  for (std::size_t i = 0; i < time_minutes.size(); ++i) {
    const SampledCurve sampled =
        sample_piecewise_linear_with_derivative(time_minutes, measured_aif, time_minutes[i] + delay);
    transformed[i] = std::max(0.0, sampled.value + tau * sampled.derivative);
  }
}

double exp_conv_step(double z, double rate, double dt, double c0, double c1) {
  if (dt <= 0.0) {
    return z;
  }

  if (std::abs(rate) < kTinyRate) {
    return z + 0.5 * (c0 + c1) * dt;
  }

  const double decay = std::exp(-rate * dt);
  const double i0 = -std::expm1(-rate * dt) / rate;

  // i1 = (1 - e^-x (1 + x)) / rate^2 with x = rate * dt. Written directly, the
  // numerator cancels to zero for small x and the division by rate^2 destroys
  // the result. Expanded in x it is
  //   i1 = dt^2 * (1/2 - x/3 + x^2/8 - x^3/30 + x^4/144 - ...),
  // which keeps full precision all the way down to rate = 0.
  const double x = rate * dt;
  const double i1 =
      std::abs(x) < kSmallRate
          ? dt * dt * (0.5 + x * (-1.0 / 3.0
                + x * (0.125 + x * (-1.0 / 30.0 + x * (1.0 / 144.0)))))
          : (1.0 - decay * (1.0 + x)) / (rate * rate);

  const double slope = (c1 - c0) / dt;
  return decay * z + c1 * i0 - slope * i1;
}

bool evaluate_reversible_fallback(
    const ParameterVector& params,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    std::vector<double>& predicted) {
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
  predicted.assign(t_count, 0.0);

  double c1 = 0.0;
  double c2 = 0.0;

  auto commit_prediction = [&](std::size_t idx, double cp) {
    const double tissue = c1 + c2;
    predicted[idx] = (1.0 - params.vB) * tissue + params.vB * cp;
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

  commit_prediction(0, aif[0]);

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

    commit_prediction(idx + 1, aif[idx + 1]);
  }

  return true;
}

}  // namespace

int active_parameter_count(ModelKind model, bool fit_vb, bool fit_delay, bool fit_dispersion) {
  return base_parameter_count(model, fit_vb) + (fit_delay ? 1 : 0) + (fit_dispersion ? 1 : 0);
}

std::string model_name(ModelKind model) {
  return model == ModelKind::Irreversible2TCM ? "2tcm-irreversible" : "2tcm-reversible";
}

std::string parameter_name(
    ModelKind model,
    bool fit_vb,
    bool fit_delay,
    bool fit_dispersion,
    int index) {
  static constexpr const char* kIrrevNoVb[] = {"K1", "k2", "k3"};
  static constexpr const char* kIrrevVb[] = {"K1", "k2", "k3", "vB"};
  static constexpr const char* kRevNoVb[] = {"K1", "k2", "k3", "k4"};
  static constexpr const char* kRevVb[] = {"K1", "k2", "k3", "k4", "vB"};

  const int n = active_parameter_count(model, fit_vb, fit_delay, fit_dispersion);
  if (index < 0 || index >= n) {
    throw std::out_of_range("Invalid parameter index");
  }

  const int base_count = base_parameter_count(model, fit_vb);
  if (index < base_count) {
    if (model == ModelKind::Irreversible2TCM) {
      return fit_vb ? kIrrevVb[index] : kIrrevNoVb[index];
    }
    return fit_vb ? kRevVb[index] : kRevNoVb[index];
  }
  int extra_index = index - base_count;
  if (fit_delay) {
    if (extra_index == 0) {
      return "delay";
    }
    --extra_index;
  }
  if (fit_dispersion && extra_index == 0) {
    return "dispersion";
  }
  throw std::out_of_range("Invalid parameter index");
}

SolverBounds default_bounds(
    ModelKind model,
    bool fit_vb,
    double fixed_vb,
    bool fit_delay,
    double fixed_delay,
    bool fit_dispersion,
    double fixed_dispersion) {
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
  if (model == ModelKind::Reversible2TCM) {
    bounds.lower[index] = 0.0;
    bounds.upper[index++] = 1.0;  // k4
  }
  if (fit_vb) {
    bounds.lower[index] = 0.0;
    bounds.upper[index++] = 1.0;
  } else {
    (void)fixed_vb;
  }
  if (fit_delay) {
    bounds.lower[index] = -0.2;
    bounds.upper[index++] = 0.2;
  } else {
    (void)fixed_delay;
  }
  if (fit_dispersion) {
    bounds.lower[index] = 0.0;
    bounds.upper[index++] = 0.1;
  } else {
    (void)fixed_dispersion;
  }
  return bounds;
}

ParameterVector default_initial_guess(
    ModelKind model,
    bool fit_vb,
    double fixed_vb,
    bool fit_delay,
    double fixed_delay,
    bool fit_dispersion,
    double fixed_dispersion) {
  ParameterVector params{};
  params.K1 = 0.1;
  params.k2 = 0.1;
  params.k3 = 0.1;
  params.k4 = model == ModelKind::Reversible2TCM ? 0.05 : 0.0;
  params.vB = fit_vb ? 0.05 : fixed_vb;
  params.delay = fit_delay ? 0.0 : fixed_delay;
  params.dispersion = fit_dispersion ? 0.02 : fixed_dispersion;
  return params;
}

ParameterVector clip_to_bounds(
    const ParameterVector& params,
    ModelKind model,
    bool fit_vb,
    double fixed_vb,
    bool fit_delay,
    double fixed_delay,
    bool fit_dispersion,
    double fixed_dispersion,
    const SolverBounds& bounds) {
  std::array<double, kMaxParameterCount> values{};
  pack_active(params, model, fit_vb, fit_delay, fit_dispersion, values);

  const int n = active_parameter_count(model, fit_vb, fit_delay, fit_dispersion);
  for (int i = 0; i < n; ++i) {
    values[static_cast<std::size_t>(i)] = std::clamp(
        values[static_cast<std::size_t>(i)],
        bounds.lower[static_cast<std::size_t>(i)],
        bounds.upper[static_cast<std::size_t>(i)]);
  }

  return unpack_active(
      values,
      model,
      fit_vb,
      fixed_vb,
      fit_delay,
      fixed_delay,
      fit_dispersion,
      fixed_dispersion);
}

void pack_active(
    const ParameterVector& params,
    ModelKind model,
    bool fit_vb,
    bool fit_delay,
    bool fit_dispersion,
    std::array<double, kMaxParameterCount>& values) {
  values.fill(0.0);
  std::size_t index = 0;
  values[index++] = params.K1;
  values[index++] = params.k2;
  values[index++] = params.k3;
  if (model == ModelKind::Reversible2TCM) {
    values[index++] = params.k4;
    if (fit_vb) {
      values[index++] = params.vB;
    }
  } else if (fit_vb) {
    values[index++] = params.vB;
  }
  index = append_if_active(fit_delay, params.delay, values, index);
  (void)append_if_active(fit_dispersion, params.dispersion, values, index);
}

ParameterVector unpack_active(
    const std::array<double, kMaxParameterCount>& values,
    ModelKind model,
    bool fit_vb,
    double fixed_vb,
    bool fit_delay,
    double fixed_delay,
    bool fit_dispersion,
    double fixed_dispersion) {
  ParameterVector params{};
  std::size_t index = 0;
  params.K1 = values[index++];
  params.k2 = values[index++];
  params.k3 = values[index++];
  params.k4 = model == ModelKind::Reversible2TCM ? values[index++] : 0.0;
  params.vB = fit_vb ? values[index++] : fixed_vb;
  params.delay = fit_delay ? values[index++] : fixed_delay;
  params.dispersion = fit_dispersion ? values[index++] : fixed_dispersion;
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

std::vector<double> default_frame_weights(const std::vector<double>& time_minutes) {
  const std::size_t n = time_minutes.size();
  std::vector<double> weights(n, 1.0);
  if (n < 2) {
    return weights;
  }

  std::vector<double> widths(n, 0.0);
  widths[0] = std::max(0.0, time_minutes[1] - time_minutes[0]);
  widths[n - 1] = std::max(0.0, time_minutes[n - 1] - time_minutes[n - 2]);
  for (std::size_t i = 1; i + 1 < n; ++i) {
    widths[i] = 0.5 * std::max(0.0, time_minutes[i + 1] - time_minutes[i - 1]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    weights[i] = std::sqrt(std::max(widths[i], 1e-6));
  }
  return weights;
}

bool evaluate_model(
    ModelKind model,
    const ParameterVector& raw_params,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif_raw,
    std::vector<double>& predicted) {
  if (time_minutes.size() != aif_raw.size() || time_minutes.empty()) {
    return false;
  }

  ParameterVector params = raw_params;
  params.vB = clamp01(params.vB);
  params.dispersion = std::max(0.0, params.dispersion);
  if (params.K1 < 0.0 || params.k2 < 0.0 || params.k3 < 0.0 || params.k4 < 0.0) {
    return false;
  }

  std::vector<double> aif = aif_raw;
  for (double& value : aif) {
    if (value < 1e-5) {
      value = 0.0;
    }
  }
  if (std::abs(params.delay) > 1e-12 || params.dispersion > 1e-12) {
    transform_aif(time_minutes, aif, params.delay, params.dispersion, aif);
  }

  const std::size_t t_count = time_minutes.size();
  predicted.assign(t_count, 0.0);

  if (model == ModelKind::Irreversible2TCM) {
    const double lambda = std::max(params.k2 + params.k3, kTinyRate);
    double z = 0.0;
    double i_cp = 0.0;

    if (time_minutes[0] > 0.0) {
      z = exp_conv_step(0.0, lambda, time_minutes[0], 0.0, aif[0]);
      i_cp = 0.5 * aif[0] * time_minutes[0];
    }

    const double tissue0 =
        params.K1 / lambda * (params.k3 * i_cp + params.k2 * z);
    predicted[0] = (1.0 - params.vB) * tissue0 + params.vB * aif[0];

    for (std::size_t idx = 0; idx + 1 < t_count; ++idx) {
      const double dt = time_minutes[idx + 1] - time_minutes[idx];
      if (dt <= 0.0) {
        return false;
      }
      z = exp_conv_step(z, lambda, dt, aif[idx], aif[idx + 1]);
      i_cp += 0.5 * (aif[idx] + aif[idx + 1]) * dt;
      const double tissue = params.K1 / lambda * (params.k3 * i_cp + params.k2 * z);
      predicted[idx + 1] = (1.0 - params.vB) * tissue + params.vB * aif[idx + 1];
    }

    return true;
  }

  const double rate_sum = params.k2 + params.k3 + params.k4;
  const double discriminant =
      std::max(0.0, rate_sum * rate_sum - 4.0 * params.k2 * params.k4);
  const double root = std::sqrt(discriminant);
  const double a1 = 0.5 * (rate_sum + root);
  const double a2 = 0.5 * (rate_sum - root);
  const double denom = a1 - a2;

  if (std::abs(denom) < 1e-8) {
    return evaluate_reversible_fallback(params, time_minutes, aif, predicted);
  }

  const double c1 = params.K1 * (a1 - params.k3 - params.k4) / denom;
  const double c2 = -params.K1 * (a2 - params.k3 - params.k4) / denom;

  double z1 = 0.0;
  double z2 = 0.0;
  if (time_minutes[0] > 0.0) {
    z1 = exp_conv_step(0.0, a1, time_minutes[0], 0.0, aif[0]);
    z2 = exp_conv_step(0.0, a2, time_minutes[0], 0.0, aif[0]);
  }

  predicted[0] = (1.0 - params.vB) * (c1 * z1 + c2 * z2) + params.vB * aif[0];

  for (std::size_t idx = 0; idx + 1 < t_count; ++idx) {
    const double dt = time_minutes[idx + 1] - time_minutes[idx];
    if (dt <= 0.0) {
      return false;
    }

    z1 = exp_conv_step(z1, a1, dt, aif[idx], aif[idx + 1]);
    z2 = exp_conv_step(z2, a2, dt, aif[idx], aif[idx + 1]);
    predicted[idx + 1] =
        (1.0 - params.vB) * (c1 * z1 + c2 * z2) + params.vB * aif[idx + 1];
  }

  return true;
}

}  // namespace tcm
