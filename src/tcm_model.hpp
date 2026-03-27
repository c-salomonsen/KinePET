#pragma once

#include <array>
#include <string>
#include <vector>

namespace tcm {

enum class ModelKind {
  Irreversible2TCM,
  Reversible2TCM,
};

struct ParameterVector {
  double K1 = 0.1;
  double k2 = 0.1;
  double k3 = 0.1;
  double k4 = 0.05;
  double vB = 0.05;
};

struct SolverBounds {
  std::array<double, 5> lower{};
  std::array<double, 5> upper{};
};

int active_parameter_count(ModelKind model, bool fit_vb);
std::string model_name(ModelKind model);
std::string parameter_name(ModelKind model, bool fit_vb, int index);
SolverBounds default_bounds(ModelKind model, bool fit_vb, double fixed_vb);
ParameterVector default_initial_guess(ModelKind model, bool fit_vb, double fixed_vb);
ParameterVector clip_to_bounds(
    const ParameterVector& params,
    ModelKind model,
    bool fit_vb,
    double fixed_vb,
    const SolverBounds& bounds);
void pack_active(
    const ParameterVector& params,
    ModelKind model,
    bool fit_vb,
    std::array<double, 5>& values);
ParameterVector unpack_active(
    const std::array<double, 5>& values,
    ModelKind model,
    bool fit_vb,
    double fixed_vb);
double compute_ki(const ParameterVector& params);
std::vector<double> default_frame_weights(const std::vector<double>& time_minutes);
bool evaluate_model(
    ModelKind model,
    const ParameterVector& params,
    const std::vector<double>& time_minutes,
    const std::vector<double>& aif,
    std::vector<double>& predicted);

}  // namespace tcm
