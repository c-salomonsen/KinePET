#pragma once

#include <string>
#include <variant>

namespace tcm {

// Vascular contribution is independent of tissue kinetics. FixedVB{0.0}
// disables the contribution; FixedVB{0.05} retains a known contribution.
struct FittedVB {};
struct FixedVB { double value = 0.05; };
using BloodFraction = std::variant<FittedVB, FixedVB>;

struct Irreversible2TCM { BloodFraction vb = FittedVB{}; };
struct Reversible2TCM { BloodFraction vb = FittedVB{}; };
using Model = std::variant<Irreversible2TCM, Reversible2TCM>;

// Translate legacy string/boolean interfaces once, at the application boundary.
Model make_model(const std::string& name, BloodFraction vb = FittedVB{});
BloodFraction blood_fraction_setting(bool fitted, double fixed_value);
const BloodFraction& blood_fraction(const Model& model);
bool fits_vb(const Model& model);
double fixed_vb(const Model& model);
bool is_reversible(const Model& model);
std::string model_name(const Model& model);

// Delay and dispersion remain independent optional AIF fit parameters.
struct ParameterSetting {
  bool fitted = false;
  double fixed_value = 0.0;
};
struct AifOptions {
  ParameterSetting delay;
  ParameterSetting dispersion;
};

}  // namespace tcm
