from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

import kinepet


def exp_conv_step(z: float, rate: float, dt: float, c0: float, c1: float) -> float:
    if dt <= 0.0:
        return z
    if abs(rate) < 1e-12:
        return z + 0.5 * (c0 + c1) * dt
    decay = np.exp(-rate * dt)
    i0 = (1.0 - decay) / rate
    i1 = (1.0 - decay * (1.0 + rate * dt)) / (rate**2)
    slope = (c1 - c0) / dt
    return decay * z + c1 * i0 - slope * i1


def forward_irrev(
    time: np.ndarray,
    aif: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    vb: float,
) -> np.ndarray:
    lam = max(k2 + k3, 1e-12)
    z = 0.0
    integral_cp = 0.0
    out = np.zeros_like(time)
    if time[0] > 0.0:
        z = exp_conv_step(0.0, lam, float(time[0]), 0.0, float(aif[0]))
        integral_cp = 0.5 * float(aif[0]) * float(time[0])
    out[0] = (1.0 - vb) * (k1 / lam * (k3 * integral_cp + k2 * z)) + vb * aif[0]
    for i in range(len(time) - 1):
        dt = float(time[i + 1] - time[i])
        z = exp_conv_step(z, lam, dt, float(aif[i]), float(aif[i + 1]))
        integral_cp += 0.5 * float(aif[i] + aif[i + 1]) * dt
        out[i + 1] = (1.0 - vb) * (k1 / lam * (k3 * integral_cp + k2 * z)) + vb * aif[i + 1]
    return out


def make_time() -> np.ndarray:
    return np.array(
        [0.0, 0.0833, 0.1667, 0.25, 0.3333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0],
        dtype=np.float64,
    )


def test_fit_tacs_recovers_irreversible_batch_with_per_tac_aif() -> None:
    time = make_time()
    time_2 = time * 1.05
    time_matrix = np.stack([time, time_2], axis=1)
    aif = np.stack(
        [
            12.0 * np.exp(-time / 0.3) + 1.5 * np.exp(-time / 6.0),
            8.5 * np.exp(-time_2 / 0.45) + 2.2 * np.exp(-time_2 / 8.0),
        ],
        axis=1,
    )
    truth = {
        "K1": np.array([0.10, 0.14], dtype=np.float64),
        "k2": np.array([0.12, 0.18], dtype=np.float64),
        "k3": np.array([0.06, 0.08], dtype=np.float64),
        "vB": np.array([0.04, 0.06], dtype=np.float64),
    }
    tacs = np.stack(
        [
            forward_irrev(time, aif[:, 0], truth["K1"][0], truth["k2"][0], truth["k3"][0], truth["vB"][0]),
            forward_irrev(time_2, aif[:, 1], truth["K1"][1], truth["k2"][1], truth["k3"][1], truth["vB"][1]),
        ],
        axis=1,
    )

    fit = kinepet.fit_tacs(tacs, time_matrix, aif, model="irr", threads=2)

    assert len(fit) == 2
    assert fit.fitted_tacs == 2
    assert_allclose(fit.K1, truth["K1"], atol=5e-3)
    assert_allclose(fit.k2, truth["k2"], atol=5e-3)
    assert_allclose(fit.k3, truth["k3"], atol=5e-3)
    assert_allclose(fit.vB, truth["vB"], atol=5e-3)
    assert fit.k4 is None
    assert fit.status_code.shape == fit.status.shape
    assert all(isinstance(value, str) for value in fit.status.tolist())
    assert np.all(np.isfinite(fit.rmse))


def test_fit_tacs_interprets_long_scan_times_as_minutes() -> None:
    time = np.array(
        [0.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0, 70.0, 90.0, 120.0],
        dtype=np.float64,
    )
    aif = 9.0 * np.exp(-time / 4.0) + 2.0 * np.exp(-time / 45.0)
    truth = {
        "K1": 0.11,
        "k2": 0.13,
        "k3": 0.04,
        "vB": 0.05,
    }
    tac = kinepet.evaluate_model(time, aif, model="irr", **truth)
    tacs = tac[:, None]

    fit = kinepet.fit_tacs(tacs, time, aif, model="irr", max_iterations=120)

    assert_allclose(fit.K1[0], truth["K1"], atol=5e-3)
    assert_allclose(fit.k2[0], truth["k2"], atol=5e-3)
    assert_allclose(fit.k3[0], truth["k3"], atol=5e-3)
    assert_allclose(fit.vB[0], truth["vB"], atol=5e-3)
    assert_allclose(
        kinepet.default_frame_weights(time),
        np.sqrt([2.0, 2.5, 4.0, 7.5, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0]),
    )


def test_fit_tacs_requires_per_tac_time_for_per_tac_aif() -> None:
    time = make_time()
    tacs = np.zeros((time.size, 2), dtype=np.float64)
    aif = np.zeros((time.size, 2), dtype=np.float64)

    try:
        kinepet.fit_tacs(tacs, time, aif, model="irr")
    except ValueError as exc:
        assert "time must have shape (T, N)" in str(exc)
    else:
        raise AssertionError("Expected fit_tacs() to require per-TAC time for per-TAC AIFs")


def test_fit_one_tac_and_evaluate_model_roundtrip_reversible() -> None:
    time = make_time()
    aif = 10.0 * np.exp(-time / 0.35) + 1.8 * np.exp(-time / 7.0)
    truth = {
        "K1": 0.12,
        "k2": 0.16,
        "k3": 0.09,
        "k4": 0.04,
        "vB": 0.05,
        "delay": 0.0,
        "dispersion": 0.0,
    }
    tac = kinepet.evaluate_model(
        time,
        aif,
        model="rev",
        **truth,
    )

    fit = kinepet.fit_one_tac(
        tac,
        time,
        aif,
        model="rev",
        max_iterations=200,
    )

    assert fit.k4 is not None
    assert isinstance(fit.status, str)
    assert isinstance(fit.status_code, int)
    assert abs(fit.K1 - truth["K1"]) < 1e-2
    assert abs(fit.k2 - truth["k2"]) < 1e-2
    assert abs(fit.k3 - truth["k3"]) < 1e-2
    assert abs(fit.k4 - truth["k4"]) < 1e-2
    assert abs(fit.vB - truth["vB"]) < 1e-2

    dispersed = kinepet.evaluate_model(
        time,
        aif,
        model="rev",
        K1=truth["K1"],
        k2=truth["k2"],
        k3=truth["k3"],
        k4=truth["k4"],
        vB=truth["vB"],
        delay=0.03,
        dispersion=0.02,
    )
    assert dispersed.shape == time.shape


def test_default_bounds_include_optional_parameters() -> None:
    bounds = kinepet.default_bounds(
        model="rev",
        fit_vb=True,
        fit_delay=True,
        fit_dispersion=True,
    )
    assert "K1" in bounds
    assert "k4" in bounds
    assert "vB" in bounds
    assert "delay" in bounds
    assert "dispersion" in bounds


def test_status_code_to_name() -> None:
    assert kinepet.status_code_to_name(1) == "gradient_converged"
    assert kinepet.status_code_to_name(4) == "max_iterations"


def test_fit_tacs_rejects_invalid_aif_columns() -> None:
    time = make_time()
    tacs = np.zeros((time.size, 3), dtype=np.float64)
    bad_aif = np.zeros((time.size, 2), dtype=np.float64)

    try:
        kinepet.fit_tacs(tacs, time, bad_aif, model="irr")
    except ValueError as exc:
        assert "aif second dimension" in str(exc)
    else:
        raise AssertionError("Expected fit_tacs() to reject invalid aif shape")
