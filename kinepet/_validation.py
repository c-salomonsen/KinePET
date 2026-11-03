"""Array normalization and shared/per-TAC shape validation."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

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


