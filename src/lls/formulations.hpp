#pragma once

#include "regression.hpp"

namespace tcm::lls {

// Model-specific dispatch; regression and recovery live beside each model.
bool estimate_linear_parameters(const LinearInputs&, const Irreversible2TCM&, ParameterVector&);
bool estimate_linear_parameters(const LinearInputs&, const Reversible2TCM&, ParameterVector&);

// Recovery is separate from regression so the algebra can be checked directly,
// independently of sampling, quadrature, regularization, and nonlinear fitting.
bool recover_irreversible_fitted_vb(const Coefficients&, ParameterVector&);
bool recover_irreversible_fixed_vb(const Coefficients&, ParameterVector&);
bool recover_reversible_fitted_vb(const Coefficients&, ParameterVector&);
bool recover_reversible_fixed_vb(const Coefficients&, ParameterVector&);

}  // namespace tcm::lls
