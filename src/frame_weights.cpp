#include "frame_weights.hpp"

#include <algorithm>
#include <cmath>

namespace tcm {

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

}  // namespace tcm
