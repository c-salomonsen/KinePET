"""Python fitting entrypoints; batch and nonlinear work are delegated to C++."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import _core
from ._validation import _as_float64_array, _as_float64_array_with_ndims, _validate_tac_inputs
from .models import Model, _resolve_fit_model
from .results import BatchFitResult, SingleTacFitResult, _status_codes_to_names, status_code_to_name

def fit_tacs(
    tacs: npt.ArrayLike,
    time: npt.ArrayLike,
    aif: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    model: Model = "rev",
    fit_vb: bool | None = None,
    fixed_vb: float | None = None,
    fit_delay: bool = False,
    fixed_delay: float = 0.0,
    fit_dispersion: bool = False,
    fixed_dispersion: float = 0.0,
    threads: int = 0,
    max_iterations: int = 40,
    gradient_tolerance: float = 1e-6,
    step_tolerance: float = 1e-6,
    cost_tolerance: float = 1e-9,
    fd_step: float = 1e-4,
    lambda_initial: float = 1e-2,
    use_linear_warm_start: bool = True,
) -> BatchFitResult:
    """Fit a flattened batch of TACs with a two-tissue compartment model.

    Parameters
    ----------
    tacs
        TAC matrix with shape ``(T, N)``. Each column is one TAC.
    time
        Frame sample times with shape ``(T,)``, ``(T, 1)``, or ``(T, N)``.
        Values are always interpreted directly as minutes. KinePET does not
        infer seconds or rescale large time values, so callers must convert
        seconds to minutes before fitting.
    aif
        Arterial input function with shape ``(T,)``, ``(T, 1)``, or
        ``(T, N)``. A vector or single-column array is shared by every TAC.
        A ``(T, N)`` array provides one AIF per TAC and requires ``time`` to
        also be ``(T, N)``.
    weights
        Optional residual weights with the same shared or per-TAC shape
        semantics as ``aif``.
    model
        ``Reversible2TCM(vb=...)`` or ``Irreversible2TCM(vb=...)``.
        Legacy names ``"rev"`` and ``"irr"`` remain supported.
    fit_vb, fixed_vb
        Legacy string-model options (defaults: True and 0.05). For a model
        object, configure ``FittedVB()`` or ``FixedVB(value)`` on the object;
        passing these legacy options as well raises ValueError.
    fit_delay, fixed_delay
        Fit AIF delay in minutes, or use ``fixed_delay`` when disabled.
    fit_dispersion, fixed_dispersion
        Fit AIF dispersion in minutes, or use ``fixed_dispersion`` when
        disabled.
    threads
        OpenMP thread count when available. ``0`` lets the backend choose.

    Returns
    -------
    BatchFitResult
        Flat per-TAC fitted parameters and diagnostics with shape ``(N,)``.
    """
    tacs_array = _as_float64_array(tacs, ndim=2, name="tacs")
    time_array = _as_float64_array_with_ndims(time, ndims=(1, 2), name="time")
    aif_array = np.asarray(aif, dtype=np.float64, order="C")
    weights_array = None if weights is None else np.asarray(weights, dtype=np.float64, order="C")
    _validate_tac_inputs(tacs_array, time_array, aif_array, weights_array)

    model_name, fit_vb, fixed_vb = _resolve_fit_model(model, fit_vb, fixed_vb)
    raw = _core.fit_tacs(
        tacs_array,
        time_array,
        aif_array,
        weights_array,
        model_name,
        fit_vb,
        fixed_vb,
        fit_delay,
        fixed_delay,
        fit_dispersion,
        fixed_dispersion,
        threads,
        max_iterations,
        gradient_tolerance,
        step_tolerance,
        cost_tolerance,
        fd_step,
        lambda_initial,
        use_linear_warm_start,
    )
    raw["status_code"] = np.asarray(raw["status"], dtype=np.int32)
    raw["status"] = _status_codes_to_names(raw["status_code"])
    return BatchFitResult(**raw)


def fit_one_tac(
    tac: npt.ArrayLike,
    time: npt.ArrayLike,
    aif: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    model: Model = "rev",
    fit_vb: bool | None = None,
    fixed_vb: float | None = None,
    fit_delay: bool = False,
    fixed_delay: float = 0.0,
    fit_dispersion: bool = False,
    fixed_dispersion: float = 0.0,
    max_iterations: int = 40,
    gradient_tolerance: float = 1e-6,
    step_tolerance: float = 1e-6,
    cost_tolerance: float = 1e-9,
    fd_step: float = 1e-4,
    lambda_initial: float = 1e-2,
    use_linear_warm_start: bool = True,
) -> SingleTacFitResult:
    """Fit one TAC with a two-tissue compartment model.

    ``tac``, ``time``, and ``aif`` must be one-dimensional arrays of equal
    length. ``time`` is always interpreted directly as minutes; KinePET does
    not infer seconds from large values or perform unit conversion. Convert
    seconds to minutes before calling this function. Model objects own their
    blood-fraction setting; do not also pass ``fit_vb`` or ``fixed_vb``.
    """
    tac_array = _as_float64_array(tac, ndim=1, name="tac")
    time_array = _as_float64_array(time, ndim=1, name="time")
    aif_array = _as_float64_array(aif, ndim=1, name="aif")
    if tac_array.shape[0] != time_array.shape[0] or aif_array.shape[0] != time_array.shape[0]:
        raise ValueError("tac, time, and aif must have the same length")
    weights_array = None
    if weights is not None:
        weights_array = _as_float64_array(weights, ndim=1, name="weights")
        if weights_array.shape[0] != time_array.shape[0]:
            raise ValueError("weights must have the same length as time")

    model_name, fit_vb, fixed_vb = _resolve_fit_model(model, fit_vb, fixed_vb)
    raw = _core.fit_one_tac(
        tac_array,
        time_array,
        aif_array,
        weights_array,
        model_name,
        fit_vb,
        fixed_vb,
        fit_delay,
        fixed_delay,
        fit_dispersion,
        fixed_dispersion,
        max_iterations,
        gradient_tolerance,
        step_tolerance,
        cost_tolerance,
        fd_step,
        lambda_initial,
        use_linear_warm_start,
    )
    raw["status_code"] = int(raw["status"])
    raw["status"] = status_code_to_name(raw["status_code"])
    return SingleTacFitResult(**raw)


