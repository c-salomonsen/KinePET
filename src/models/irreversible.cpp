#include "tissue.hpp"
#include "convolution.hpp"

#include <algorithm>

namespace tcm::kinetics {

// q' = K1*C_A-k2*f, with k4=0. The impulse response is
// K1/(k2+k3) * [k3 + k2*exp(-(k2+k3)*t)].
bool evaluate_tissue(const Irreversible2TCM&, const ParameterVector& params,
                     const std::vector<double>& time_minutes,
                     const std::vector<double>& aif, std::vector<double>& tissue) {
  constexpr double kTinyRate = 1e-12;
  const std::size_t t_count = time_minutes.size();
  tissue.assign(t_count, 0.0);
  const double lambda = std::max(params.k2 + params.k3, kTinyRate);
  double z = 0.0;
  double i_cp = 0.0;

  if (time_minutes[0] > 0.0) {
    z = exp_conv_step(0.0, lambda, time_minutes[0], 0.0, aif[0]);
    i_cp = 0.5 * aif[0] * time_minutes[0];
  }

  const double tissue0 =
      params.K1 / lambda * (params.k3 * i_cp + params.k2 * z);
  tissue[0] = tissue0;

  for (std::size_t idx = 0; idx + 1 < t_count; ++idx) {
    const double dt = time_minutes[idx + 1] - time_minutes[idx];
    if (dt <= 0.0) {
      return false;
    }
    z = exp_conv_step(z, lambda, dt, aif[idx], aif[idx + 1]);
    i_cp += 0.5 * (aif[idx] + aif[idx + 1]) * dt;
    const double concentration = params.K1 / lambda * (params.k3 * i_cp + params.k2 * z);
    tissue[idx + 1] = concentration;
  }

  return true;
}

}  // namespace tcm::kinetics
