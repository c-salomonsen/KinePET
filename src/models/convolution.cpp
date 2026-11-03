#include "convolution.hpp"

#include <cmath>

namespace tcm::kinetics {
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

}  // namespace

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

}  // namespace tcm::kinetics
