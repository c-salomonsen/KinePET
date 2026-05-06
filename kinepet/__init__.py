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


def _validate_tac_inputs(
    tacs: np.ndarray,
    time: np.ndarray,
    aif: np.ndarray,
    weights: np.ndarray | None,
) -> None:
    if tacs.ndim != 2:
        raise ValueError(f"tacs must have shape (T, N), got {tacs.shape}")
    if time.ndim != 1:
        raise ValueError(f"time must have shape (T,), got {time.shape}")
    if aif.ndim not in (1, 2):
        raise ValueError(f"aif must have shape (T,), (T, 1), or (T, N), got {aif.shape}")
    if tacs.shape[0] != time.shape[0]:
        raise ValueError("time length must match the first dimension of tacs")
    if aif.shape[0] != time.shape[0]:
        raise ValueError("aif first dimension must match the time vector length")
    if aif.ndim == 2 and aif.shape[1] not in (1, tacs.shape[1]):
        raise ValueError("aif second dimension must be 1 or match the number of TACs")
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
    tacs_array = _as_float64_array(tacs, ndim=2, name="tacs")
    time_array = _as_float64_array(time, ndim=1, name="time")
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


default_bounds = _core.default_bounds
default_frame_weights = _core.default_frame_weights
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
