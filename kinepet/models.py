# SPDX-FileCopyrightText: 2026 Christian Salomonsen
# SPDX-License-Identifier: AGPL-3.0-only

"""Kinetic model descriptions; all numerical kinetics remain in compiled C++."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, TypeAlias, cast

ModelName = Literal["irr", "rev"]


@dataclass(frozen=True, slots=True)
class FittedVB:
    """Estimate the vascular blood fraction together with the kinetic rates."""


@dataclass(frozen=True, slots=True)
class FixedVB:
    """Use a known blood fraction. ``FixedVB(0.0)`` removes vascular activity."""

    value: float = 0.05

    def __post_init__(self) -> None:
        if not isfinite(self.value) or not 0.0 <= self.value <= 1.0:
            raise ValueError("FixedVB.value must be finite and between 0 and 1")


BloodFraction: TypeAlias = FittedVB | FixedVB


def _validate_blood_fraction(vb: BloodFraction) -> None:
    if not isinstance(vb, (FittedVB, FixedVB)):
        raise TypeError("vb must be FittedVB() or FixedVB(value)")


@dataclass(frozen=True, slots=True)
class Irreversible2TCM:
    """Irreversible two-tissue kinetics (K1, k2, k3), with explicit blood fraction."""

    vb: BloodFraction = field(default_factory=FittedVB)

    def __post_init__(self) -> None:
        _validate_blood_fraction(self.vb)


@dataclass(frozen=True, slots=True)
class Reversible2TCM:
    """Reversible two-tissue kinetics (K1, k2, k3, k4), with explicit blood fraction."""

    vb: BloodFraction = field(default_factory=FittedVB)

    def __post_init__(self) -> None:
        _validate_blood_fraction(self.vb)


KineticModel: TypeAlias = Irreversible2TCM | Reversible2TCM
Model: TypeAlias = ModelName | KineticModel


def _model_name(model: Model) -> ModelName:
    if isinstance(model, Irreversible2TCM):
        return "irr"
    if isinstance(model, Reversible2TCM):
        return "rev"
    if isinstance(model, str):
        return cast(ModelName, model)  # Preserve compiled legacy string validation.
    raise TypeError("model must be 'irr', 'rev', Irreversible2TCM, or Reversible2TCM")


def _resolve_fit_model(
    model: Model, fit_vb: bool | None, fixed_vb: float | None,
) -> tuple[ModelName, bool, float]:
    name = _model_name(model)
    if isinstance(model, str):
        return name, True if fit_vb is None else fit_vb, 0.05 if fixed_vb is None else fixed_vb
    if fit_vb is not None or fixed_vb is not None:
        raise ValueError("Set vb on the model object; do not also pass fit_vb or fixed_vb")
    if isinstance(model.vb, FixedVB):
        return name, False, model.vb.value
    return name, True, 0.05


def _resolve_evaluation_model(model: Model, vb: float | None) -> tuple[ModelName, float]:
    name = _model_name(model)
    if not isinstance(model, str) and isinstance(model.vb, FixedVB):
        if vb is not None and vb != model.vb.value:
            raise ValueError("vB conflicts with the model's FixedVB value")
        return name, model.vb.value
    return name, 0.05 if vb is None else vb
