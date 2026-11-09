// SPDX-FileCopyrightText: 2026 Christian Salomonsen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "parameters.hpp"

namespace tcm::kinetics {

// Tissue concentrations are per extravascular volume. Both overloads return
// q=f+b; vascular mixing is applied once by tcm::evaluate_model.
bool evaluate_tissue(const Irreversible2TCM&, const ParameterVector&,
                     const std::vector<double>& time_minutes,
                     const std::vector<double>& aif, std::vector<double>& tissue);
bool evaluate_tissue(const Reversible2TCM&, const ParameterVector&,
                     const std::vector<double>& time_minutes,
                     const std::vector<double>& aif, std::vector<double>& tissue);

}  // namespace tcm::kinetics
