#pragma once

#include <vector>

namespace tcm {
void transform_aif(const std::vector<double>& time_minutes,
                   const std::vector<double>& measured_aif,
                   double delay, double dispersion, std::vector<double>& transformed);
}  // namespace tcm
