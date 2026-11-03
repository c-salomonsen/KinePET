#include "model.hpp"

#include <stdexcept>

namespace tcm {

Model make_model(const std::string& name, BloodFraction vb) {
  if (name == "irr") return Irreversible2TCM{vb};
  if (name == "rev") return Reversible2TCM{vb};
  throw std::runtime_error("model must be 'rev' or 'irr'");
}

BloodFraction blood_fraction_setting(bool fitted, double fixed_value) {
  if (fitted) return FittedVB{};
  return FixedVB{fixed_value};
}

const BloodFraction& blood_fraction(const Model& model) {
  return std::visit([](const auto& kinetics) -> const BloodFraction& {
    return kinetics.vb;
  }, model);
}

bool fits_vb(const Model& model) {
  return std::holds_alternative<FittedVB>(blood_fraction(model));
}

double fixed_vb(const Model& model) {
  const auto* fixed = std::get_if<FixedVB>(&blood_fraction(model));
  return fixed ? fixed->value : 0.05;
}

bool is_reversible(const Model& model) {
  return std::holds_alternative<Reversible2TCM>(model);
}

std::string model_name(const Model& model) {
  return is_reversible(model) ? "2tcm-reversible" : "2tcm-irreversible";
}

}  // namespace tcm
