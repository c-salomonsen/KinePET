#pragma once

namespace tcm::kinetics {
// Advance integral_0^t C_A(u)*exp(-rate*(t-u)) du over a linear AIF segment.
double exp_conv_step(double previous, double rate, double dt, double c0, double c1);
}  // namespace tcm::kinetics
