# SPDX-FileCopyrightText: 2026 Christian Salomonsen
# SPDX-License-Identifier: AGPL-3.0-only

"""Invalid inputs must never masquerade as convergence, including via _core."""
import numpy as np
import pytest

import kinepet
from kinepet import _core


@pytest.mark.parametrize("field", ["tacs", "time", "aif", "weights"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_are_invalid(field, value):
    inputs = dict(tacs=np.ones((6, 1)), time=np.arange(6.),
                  aif=np.ones(6), weights=np.ones(6))
    inputs[field][2] = value
    result = _core.fit_tacs(**inputs)
    assert result["status"].tolist() == [5]
    assert np.isnan(result["rmse"]).all()
    assert np.isnan(result["weighted_cost"]).all()


@pytest.mark.parametrize("time", [[0, 1, 1, 2], [0, 2, 1, 3], [-1, 0, 1, 2]])
def test_invalid_time_order(time):
    fit = kinepet.fit_one_tac(np.ones(4), time, np.ones(4))
    assert fit.status == "invalid_input"


@pytest.mark.parametrize("weights", [np.zeros(6), [-1, 1, 1, 1, 1, 1]])
def test_invalid_weights(weights):
    fit = kinepet.fit_one_tac(np.ones(6), np.arange(6.), np.ones(6), weights=weights)
    assert fit.status_code == 5


@pytest.mark.parametrize("setting", [
    {"fd_step": 0}, {"fd_step": np.nan}, {"lambda_initial": np.inf},
    {"lambda_initial": -1}, {"max_iterations": -1}, {"threads": -1},
    {"gradient_tolerance": np.nan}, {"cost_tolerance": -1},
    {"step_tolerance": np.inf}, {"fixed_delay": np.nan},
    {"fixed_dispersion": np.inf}, {"fit_vb": False, "fixed_vb": np.nan},
    {"fit_vb": False, "fixed_vb": 1.1},
])
def test_invalid_solver_options_raise(setting):
    with pytest.raises((ValueError, RuntimeError)):
        kinepet.fit_tacs(np.ones((6, 1)), np.arange(6.), np.ones(6), **setting)


def test_mixed_batch_preserves_valid_column():
    time = np.linspace(0, 30, 50)
    aif = 10 * np.exp(-time) + np.exp(-time / 10)
    tac = kinepet.evaluate_model(time, aif, model="irr", K1=.1, k2=.2, k3=.1)
    fit = kinepet.fit_tacs(np.column_stack([tac, np.full_like(tac, np.nan)]), time, aif, model="irr")
    assert fit.status_code[0] in (1, 2, 3)
    assert fit.status_code[1] == 5
    assert fit.fitted_tacs == 1
    assert np.isfinite(fit.rmse[0]) and np.isnan(fit.rmse[1])


def test_finite_inputs_with_overflow_do_not_converge():
    fit = kinepet.fit_one_tac(np.full(6, 1e308), np.arange(6.), np.ones(6))
    assert fit.status_code == 0
    assert np.isnan(fit.weighted_cost)


@pytest.mark.parametrize("delay", [-.03, .03])
@pytest.mark.parametrize("dispersion", [0., .02])
@pytest.mark.parametrize("model", ["irr", "rev"])
def test_pure_vascular_shift_and_dispersion(delay, dispersion, model):
    time = np.array([0., .1, .2, .5, 1., 2.])
    aif = np.array([0., 10., 5., 3., 1., .5])
    shifted = time + delay
    expected = np.interp(shifted, time, aif, left=0.)
    indices = np.clip(np.searchsorted(time, shifted, side="right") - 1, 0, len(time) - 2)
    slope = np.diff(aif)[indices] / np.diff(time)[indices]
    slope[(shifted < 0) | (shifted > time[-1])] = 0
    expected = np.maximum(0, expected + dispersion * slope)
    actual = kinepet.evaluate_model(time, aif, model=model, K1=0, k2=.2, k3=.1,
                                   k4=.05, vB=1, delay=delay, dispersion=dispersion)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("field", ["time", "aif", "K1", "delay", "dispersion"])
def test_forward_rejects_nonfinite(field):
    kwargs = dict(time=np.arange(6.), aif=np.ones(6), model="rev", K1=.1, k2=.2, k3=.1)
    kwargs[field] = np.full(6, np.nan) if field in ("time", "aif") else np.nan
    with pytest.raises(RuntimeError):
        kinepet.evaluate_model(**kwargs)
