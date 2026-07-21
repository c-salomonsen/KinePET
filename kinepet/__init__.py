from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import _core

ModelName = Literal["irr", "rev"]

_STATUS_NAMES: dict[int, str] = {
    0: "failed",
    1: "gradient_converged",
    2: "step_converged",
    3: "cost_converged",
    4: "max_iterations",
    5: "invalid_input",
}


def _as_float64_array(values: npt.ArrayLike, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64, order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}")
    return array


def _as_float64_array_with_ndims(values: npt.ArrayLike, *, ndims: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64, order="C")
    if array.ndim not in ndims:
        expected = " or ".join(f"{ndim}D" for ndim in ndims)
        raise ValueError(f"{name} must be {expected}, got shape {array.shape}")
    return array


def _validate_tac_inputs(
    tacs: np.ndarray,
    time: np.ndarray,
    aif: np.ndarray,
    weights: np.ndarray | None,
) -> None:
    if tacs.ndim != 2:
        raise ValueError(f"tacs must have shape (T, N), got {tacs.shape}")
    if time.ndim not in (1, 2):
        raise ValueError(f"time must have shape (T,), (T, 1), or (T, N), got {time.shape}")
    if aif.ndim not in (1, 2):
        raise ValueError(f"aif must have shape (T,), (T, 1), or (T, N), got {aif.shape}")
    if tacs.shape[0] != time.shape[0]:
        raise ValueError("time length must match the first dimension of tacs")
    if aif.shape[0] != time.shape[0]:
        raise ValueError("aif first dimension must match the time vector length")
    if time.ndim == 2 and time.shape[1] not in (1, tacs.shape[1]):
        raise ValueError("time second dimension must be 1 or match the number of TACs")
    if aif.ndim == 2 and aif.shape[1] not in (1, tacs.shape[1]):
        raise ValueError("aif second dimension must be 1 or match the number of TACs")
    time_is_per_tac = time.ndim == 2 and time.shape[1] == tacs.shape[1] and tacs.shape[1] > 1
    aif_is_per_tac = aif.ndim == 2 and aif.shape[1] == tacs.shape[1] and tacs.shape[1] > 1
    if aif_is_per_tac and not time_is_per_tac:
        raise ValueError("time must have shape (T, N) when aif has shape (T, N)")
    if time_is_per_tac and not aif_is_per_tac:
        raise ValueError("time must be shared when aif is shared")
    if weights is None:
        return
    if weights.ndim not in (1, 2):
        raise ValueError(f"weights must have shape (T,), (T, 1), or (T, N), got {weights.shape}")
    if weights.shape[0] != time.shape[0]:
        raise ValueError("weights first dimension must match the time vector length")
    if weights.ndim == 2 and weights.shape[1] not in (1, tacs.shape[1]):
        raise ValueError("weights second dimension must be 1 or match the number of TACs")


def status_code_to_name(status_code: int) -> str:
    return _STATUS_NAMES.get(int(status_code), f"unknown_status_{int(status_code)}")


def _status_codes_to_names(status_codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(status_codes, dtype=np.int32)
    return np.asarray([status_code_to_name(code) for code in codes], dtype=object)


@dataclass(slots=True)
class BatchFitResult:
    K1: np.ndarray
    k2: np.ndarray
    k3: np.ndarray
    vB: np.ndarray
    Ki: np.ndarray
    rmse: np.ndarray
    weighted_cost: np.ndarray
    iterations: np.ndarray
    status: np.ndarray
    status_code: np.ndarray
    fitted_tacs: int
    k4: np.ndarray | None = None
    delay: np.ndarray | None = None
    dispersion: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.K1.shape[0])

    def as_dict(self) -> dict[str, np.ndarray | int]:
        out: dict[str, np.ndarray | int] = {
            "K1": self.K1,
            "k2": self.k2,
            "k3": self.k3,
            "vB": self.vB,
            "Ki": self.Ki,
            "rmse": self.rmse,
            "weighted_cost": self.weighted_cost,
            "iterations": self.iterations,
            "status": self.status,
            "status_code": self.status_code,
            "fitted_tacs": self.fitted_tacs,
        }
        if self.k4 is not None:
            out["k4"] = self.k4
        if self.delay is not None:
            out["delay"] = self.delay
        if self.dispersion is not None:
            out["dispersion"] = self.dispersion
        return out


@dataclass(slots=True)
class SingleTacFitResult:
    K1: float
    k2: float
    k3: float
    vB: float
    Ki: float
    rmse: float
    weighted_cost: float
    iterations: int
    status: str
    status_code: int
    k4: float | None = None
    delay: float | None = None
    dispersion: float | None = None


def fit_tacs(
    tacs: npt.ArrayLike,
    time: npt.ArrayLike,
    aif: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    model: ModelName = "rev",
    fit_vb: bool = True,
    fixed_vb: float = 0.05,
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
        Model name: ``"rev"`` for reversible 2TCM or ``"irr"`` for
        irreversible 2TCM.
    fit_vb, fixed_vb
        Fit blood volume fraction, or use ``fixed_vb`` when disabled.
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

    raw = _core.fit_tacs(
        tacs_array,
        time_array,
        aif_array,
        weights_array,
        model,
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
    model: ModelName = "rev",
    fit_vb: bool = True,
    fixed_vb: float = 0.05,
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
    seconds to minutes before calling this function.
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

    raw = _core.fit_one_tac(
        tac_array,
        time_array,
        aif_array,
        weights_array,
        model,
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


def evaluate_model(
    time: npt.ArrayLike,
    aif: npt.ArrayLike,
    *,
    model: ModelName,
    K1: float,
    k2: float,
    k3: float,
    k4: float = 0.0,
    vB: float = 0.05,
    delay: float = 0.0,
    dispersion: float = 0.0,
) -> np.ndarray:
    """Evaluate a model TAC for the provided parameters.

    ``time`` and ``aif`` must be one-dimensional arrays of equal length.
    ``time``, ``delay``, and ``dispersion`` are interpreted in minutes. No
    automatic seconds-to-minutes conversion is performed.
    """
    time_array = _as_float64_array(time, ndim=1, name="time")
    aif_array = _as_float64_array(aif, ndim=1, name="aif")
    if time_array.shape[0] != aif_array.shape[0]:
        raise ValueError("time and aif must have the same length")
    return _core.evaluate_model(
        time_array,
        aif_array,
        model,
        K1,
        k2,
        k3,
        k4,
        vB,
        delay,
        dispersion,
    )


def default_bounds(
    model: ModelName = "rev",
    *,
    fit_vb: bool = True,
    fixed_vb: float = 0.05,
    fit_delay: bool = False,
    fixed_delay: float = 0.0,
    fit_dispersion: bool = False,
    fixed_dispersion: float = 0.0,
) -> dict[str, tuple[float, float]]:
    """Return default parameter bounds for a model and optional fit settings.

    Delay and dispersion bounds are expressed in minutes when those parameters
    are fitted. ``model`` must be ``"rev"`` or ``"irr"``.
    """
    return _core.default_bounds(
        model,
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


FitStatus = _core.FitStatus

__all__ = [
    "BatchFitResult",
    "FitStatus",
    "ModelName",
    "SingleTacFitResult",
    "default_bounds",
    "default_frame_weights",
    "evaluate_model",
    "fit_one_tac",
    "fit_tacs",
    "status_code_to_name",
]
