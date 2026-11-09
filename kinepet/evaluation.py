# SPDX-FileCopyrightText: 2026 Christian Salomonsen
# SPDX-License-Identifier: AGPL-3.0-only

"""Forward evaluation and parameter helpers, all backed by compiled C++."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import _core
from ._validation import _as_float64_array
from .models import Model, _resolve_evaluation_model, _resolve_fit_model

def evaluate_model(
    time: npt.ArrayLike,
    aif: npt.ArrayLike,
    *,
    model: Model,
    K1: float,
    k2: float,
    k3: float,
    k4: float = 0.0,
    vB: float | None = None,
    delay: float = 0.0,
    dispersion: float = 0.0,
) -> np.ndarray:
    """Evaluate a model TAC for the provided parameters.

    ``time`` and ``aif`` must be one-dimensional arrays of equal length.
    ``time``, ``delay``, and ``dispersion`` are interpreted in minutes. No
    automatic seconds-to-minutes conversion is performed. A model object's
    ``FixedVB`` supplies the vascular fraction; a conflicting ``vB`` raises
    ValueError. Otherwise the default vascular fraction remains 0.05.
    """
    time_array = _as_float64_array(time, ndim=1, name="time")
    aif_array = _as_float64_array(aif, ndim=1, name="aif")
    if time_array.shape[0] != aif_array.shape[0]:
        raise ValueError("time and aif must have the same length")
    model_name, vB = _resolve_evaluation_model(model, vB)
    return _core.evaluate_model(
        time_array,
        aif_array,
        model_name,
        K1,
        k2,
        k3,
        k4,
        vB,
        delay,
        dispersion,
    )


def default_bounds(
    model: Model = "rev",
    *,
    fit_vb: bool | None = None,
    fixed_vb: float | None = None,
    fit_delay: bool = False,
    fixed_delay: float = 0.0,
    fit_dispersion: bool = False,
    fixed_dispersion: float = 0.0,
) -> dict[str, tuple[float, float]]:
    """Return default parameter bounds for a model and optional fit settings.

    Delay and dispersion bounds are expressed in minutes when those parameters
    are fitted. ``model`` accepts a kinetic model object or the legacy names
    ``"rev"`` and ``"irr"``. Model objects own their blood-fraction setting.
    """
    model_name, fit_vb, fixed_vb = _resolve_fit_model(model, fit_vb, fixed_vb)
    return _core.default_bounds(
        model_name,
        fit_vb,
        fixed_vb,
        fit_delay,
        fixed_delay,
        fit_dispersion,
        fixed_dispersion,
    )


def default_frame_weights(time: npt.ArrayLike) -> np.ndarray:
    """Return default residual weights for frame sample times in minutes.

    ``time`` must be a one-dimensional vector. Values are always interpreted
    directly as minutes; KinePET does not infer seconds or rescale large time
    values.
    """
    time_array = _as_float64_array(time, ndim=1, name="time")
    return _core.default_frame_weights(time_array)

