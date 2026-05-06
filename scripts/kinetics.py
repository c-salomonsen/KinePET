from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


def trapezoidal_integral(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute cumulative trapezoidal integral of ``y`` over time ``t``.

    Args:
        y: Array with shape ``(T, N)`` where ``T`` is the number of time points
            and ``N`` is the number of voxel curves.
        t: Time points with shape ``(T,)``.

    Returns:
        Array with shape ``(T, N)`` where entry ``[i]`` is the integral from
        ``t[0]`` to ``t[i]``.
    """
    integral = np.zeros_like(y)
    dt = np.diff(t).reshape(-1, *([1] * (y.ndim - 1)))
    integral[1:, ...] = np.cumsum(0.5 * dt * (y[:-1, ...] + y[1:, ...]), axis=0)
    return integral


def exponential_convolution_trapezoidal(
    Cp: np.ndarray,
    t: np.ndarray,
    rate: np.ndarray,
) -> np.ndarray:
    """
    Compute trapezoidal approximation of:
        conv(t_i) = ∫_0^{t_i} exp(-rate * (t_i - tau)) * Cp(tau) dtau

    using the stable recurrence:
        conv_i = E_i * conv_{i-1} + 0.5 * dt_i * (E_i * Cp_{i-1} + Cp_i),
        E_i = exp(-rate * dt_i)

    Args:
        Cp: ``(T, N)`` input function values.
        t: ``(T,)`` strictly increasing time points.
        rate: ``(N,)`` non-negative exponential rates.

    Returns:
        ``(T, N)`` convolution values.
    """
    T, N = Cp.shape
    rate_row = rate.reshape(1, -1)  # (1, N)
    dt = t[1:] - t[:-1]  # (T-1,)
    conv_steps = [np.zeros(N, dtype=Cp.dtype)]

    for i in range(1, T):
        dt_i = dt[i - 1]
        E_i = np.exp(-rate_row * dt_i).reshape(-1)  # (N,)
        prev = conv_steps[-1]
        conv_i = E_i * prev + 0.5 * dt_i * (E_i * Cp[i - 1] + Cp[i])
        conv_steps.append(conv_i)

    return np.stack(conv_steps, axis=0)


class ForwardCompartmentModel(ABC):
    """
    Base class for forward dynamic PET compartment models.

    Input contract is intentionally minimal and explicit:
    - ``Cp``: ``(T,)`` arterial input curve
    - ``t``: ``(T,)`` time points (strictly increasing)
    - each parameter: scalar or flat vector ``(N,)`` where
      ``N = depth * height * width``

    Parameters are broadcast to a common ``(N,)`` shape. Each model returns:
    - ``(T,)`` for scalar parameters
    - ``(T, N)`` for vectorized parameters
    """

    parameter_names: ClassVar[tuple[str, ...]] = ()
    def __call__(self, Cp, t, **params) -> np.ndarray:
        cp = Cp.reshape(-1)

        if not np.issubdtype(cp.dtype, np.floating):
            cp = cp.astype(np.float64)

        time = t.reshape(-1)

        if cp.shape != time.shape:
            raise ValueError(f"Cp and t must have same shape, got {cp.shape} and {time.shape}.")
        if cp.size < 2:
            raise ValueError("At least two time points are required.")
        cp_finite = np.isfinite(cp)
        time_finite = np.isfinite(time)
        if not np.all(time_finite):
            bad_idx = int(np.flatnonzero(~time_finite)[0])
            bad_value = time[bad_idx].item()
            raise ValueError(
                f"t must be finite. First non-finite at index {bad_idx} with value {bad_value}."
            )
        if not np.all(cp_finite):
            bad_idx = int(np.flatnonzero(~cp_finite)[0])
            bad_value = cp[bad_idx].item()
            finite_cp = cp[cp_finite]
            if finite_cp.size > 0:
                cp_min = finite_cp.min().item()
                cp_max = finite_cp.max().item()
            else:
                cp_min = float("nan")
                cp_max = float("nan")
            raise ValueError(
                "Cp must be finite. "
                f"First non-finite at index {bad_idx} with value {bad_value}. "
                f"Finite Cp range=[{cp_min}, {cp_max}]."
            )
        if np.any(np.diff(time) <= 0):
            raise ValueError("t must be strictly increasing.")

        raw_params = {}
        for name in self.parameter_names:
            if name not in params:
                raise KeyError(f"Missing required parameter: {name}")
            value = params[name]
            if value.ndim == 0:
                value = value.reshape(1)
            elif value.ndim == 1:
                value = value.reshape(-1)
            else:
                raise ValueError(
                    f"Parameter '{name}' must be scalar or flat vector (N,), "
                    f"got shape {tuple(value.shape)}."
                )
            raw_params[name] = value

        broadcasted = np.broadcast_arrays(*(raw_params[name] for name in self.parameter_names))
        p = {name: value for name, value in zip(self.parameter_names, broadcasted)}

        ct = self.forward_cpet(cp, time, **p)
        if ct.ndim != 2 or ct.shape[0] != cp.size:
            raise RuntimeError(
                f"{self.__class__.__name__}.forward_cpet must return (T, N), got {tuple(ct.shape)}."
            )

        return ct[:, 0] if ct.shape[1] == 1 else ct

    @abstractmethod
    def forward_cpet(self, Cp: np.ndarray, t: np.ndarray, **params) -> np.ndarray:
        """
        Compute measured PET curve(s).

        Args:
            Cp: ``(T,)`` input function.
            t: ``(T,)`` time points.
            params: Broadcasted parameters, each with shape ``(N,)``.

        Returns:
            ``(T, N)`` measured curve(s).
        """
        ...


class PatlakModel(ForwardCompartmentModel):
    """
    Patlak model:
        Ct(t) = Ki * ∫Cp(τ)dτ + V0 * Cp(t)
    """

    parameter_names: ClassVar[tuple[str, ...]] = ("Ki", "V0")

    def forward_cpet(self, Cp: np.ndarray, t: np.ndarray, **params) -> np.ndarray:
        Ki, V0 = (params[name] for name in self.parameter_names)  # (N,), (N,)
        cp_2d = np.broadcast_to(Cp.reshape(-1, 1), (Cp.size, Ki.size))  # (T, N)
        integral_cp = trapezoidal_integral(cp_2d, t)  # (T, N)
        return Ki.reshape(1, -1) * integral_cp + V0.reshape(1, -1) * cp_2d


class OneTissueModelVB(ForwardCompartmentModel):
    """
    1T2K with blood-volume mixing:
        h(s) = K1 * exp(-k2 * s)
        Ct_tissue(t) = (h * Cp)(t)
        Ct(t) = (1 - vB) * Ct_tissue(t) + vB * Cp(t)
    """

    parameter_names: ClassVar[tuple[str, ...]] = ("K1", "k2", "vB")

    def forward_cpet(self, Cp: np.ndarray, t: np.ndarray, **params) -> np.ndarray:
        K1, k2, vB = (params[name] for name in self.parameter_names)  # (N,)

        cp_2d = np.broadcast_to(Cp.reshape(-1, 1), (Cp.size, K1.size))  # (T, N)

        # Ct_tissue(t) = K1 * ∫ exp(-k2*(t-tau)) * Cp(tau) dtau
        exp_convolution_term = exponential_convolution_trapezoidal(cp_2d, t, k2)
        ct_tissue = K1.reshape(1, -1) * exp_convolution_term

        return (1.0 - vB).reshape(1, -1) * ct_tissue + vB.reshape(1, -1) * cp_2d


class Irreversible2TCMVB(ForwardCompartmentModel):
    """
    Irreversible 2TCM (2T3K) with blood-volume mixing:
        h(s) = K1/(k2+k3) * (k3 + k2 * exp(-(k2+k3)s))
        Ct_tissue(t) = (h * Cp)(t)
        Ct(t) = (1 - vB) * Ct_tissue(t) + vB * Cp(t)
    """

    parameter_names: ClassVar[tuple[str, ...]] = ("K1", "k2", "k3", "vB")

    def forward_cpet(self, Cp: np.ndarray, t: np.ndarray, **params) -> np.ndarray:
        K1, k2, k3, vB = (params[name] for name in self.parameter_names)  # (N,)

        eps = np.finfo(Cp.dtype).eps
        lam = np.clip(k2 + k3, eps, None)  # (N,)

        cp_2d = np.broadcast_to(Cp.reshape(-1, 1), (Cp.size, K1.size))  # (T, N)

        # Ct_tissue = K1/lam * [k3 * ∫Cp + k2 * ∫exp(-lam*(t-tau))Cp(tau)dtau]
        integral_cp = trapezoidal_integral(cp_2d, t)
        exp_convolution_term = exponential_convolution_trapezoidal(cp_2d, t, lam)

        ct_tissue = (K1 / lam).reshape(1, -1) * (
            k3.reshape(1, -1) * integral_cp + k2.reshape(1, -1) * exp_convolution_term
        )
        return (1.0 - vB).reshape(1, -1) * ct_tissue + vB.reshape(1, -1) * cp_2d


class Reversible2TCMVB(ForwardCompartmentModel):
    """
    Reversible 2TCM (2T4K) with blood-volume mixing:
        a1, a2 = 0.5 * [(k2+k3+k4) +- sqrt((k2+k3+k4)^2 - 4*k2*k4)]
        h(s) = K1/(a1-a2) * [ (a1-k3-k4)exp(-a1 s) - (a2-k3-k4)exp(-a2 s) ]
        Ct_tissue(t) = (h * Cp)(t)
        Ct(t) = (1 - vB) * Ct_tissue(t) + vB * Cp(t)
    """

    parameter_names: ClassVar[tuple[str, ...]] = ("K1", "k2", "k3", "k4", "vB")

    def forward_cpet(self, Cp: np.ndarray, t: np.ndarray, **params) -> np.ndarray:
        K1, k2, k3, k4, vB = (params[name] for name in self.parameter_names)  # (N,)

        eps = np.finfo(Cp.dtype).eps
        s = np.clip(k2 + k3 + k4, eps, None)
        discriminant = np.clip(s * s - 4.0 * k2 * k4, 0.0, None)
        root = np.sqrt(discriminant)
        a1 = 0.5 * (s + root)  # (N,)
        a2 = 0.5 * (s - root)  # (N,)
        denom = np.clip(a1 - a2, eps, None)  # (N,)

        cp_2d = np.broadcast_to(Cp.reshape(-1, 1), (Cp.size, K1.size))  # (T, N)

        conv_a1 = exponential_convolution_trapezoidal(cp_2d, t, a1)
        conv_a2 = exponential_convolution_trapezoidal(cp_2d, t, a2)

        ct_tissue = (K1 / denom).reshape(1, -1) * (
            (a1 - k3 - k4).reshape(1, -1) * conv_a1
            - (a2 - k3 - k4).reshape(1, -1) * conv_a2
        )
        return (1.0 - vB).reshape(1, -1) * ct_tissue + vB.reshape(1, -1) * cp_2d
