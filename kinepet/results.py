"""Per-TAC result containers and stable solver status names."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core

FitStatus = _core.FitStatus

_STATUS_NAMES: dict[int, str] = {
    0: "failed",
    1: "gradient_converged",
    2: "step_converged",
    3: "cost_converged",
    4: "max_iterations",
    5: "invalid_input",
}


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


