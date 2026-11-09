"""Independent mass-balance ODE checks, used only for validation.

Each piecewise-linear AIF interval is integrated separately so the adaptive
integrator cannot miss knots. Production and synthetic-data code use _core.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import kinepet


@pytest.mark.parametrize("model", ["irr", "rev"])
@pytest.mark.parametrize("first_time", [0., .02])
@pytest.mark.parametrize("rates", [(.2, .1, .04), (1e-7, 2e-7, 1e-8), (2., .5, .4), (.1, 0., .1)])
def test_forward_matches_compartment_mass_balance(model, first_time, rates):
    time = np.array([first_time, .0833, .1667, .25, .5, 1., 2., 5., 10., 20., 30., 45.])
    aif = 12 * np.exp(-time / .3) + 1.5 * np.exp(-time / 6)
    K1, vb = .12, .05
    k2, k3, k4 = rates
    if model == "irr":
        k4 = 0.
    state = np.zeros(2)
    reference = []
    left, cp_left = 0., 0. if first_time else aif[0]
    for right, cp_right in zip(time, aif):
        if right > left:
            def derivative(t, c):
                cp = cp_left + (cp_right - cp_left) * (t - left) / (right - left)
                return [K1 * cp - (k2 + k3) * c[0] + k4 * c[1], k3 * c[0] - k4 * c[1]]
            solution = solve_ivp(derivative, (left, right), state, method="DOP853", rtol=2e-12, atol=2e-14)
            assert solution.success
            state = solution.y[:, -1]
        reference.append((1 - vb) * state.sum() + vb * cp_right)
        left, cp_left = right, cp_right
    actual = kinepet.evaluate_model(time, aif, model=model, K1=K1, k2=k2, k3=k3, k4=k4, vB=vb)
    # The repeated-root branch uses eight RK4 substeps per interval.
    tolerance = 3e-5 if model == "rev" and rates == (.1, 0., .1) else 2e-9
    np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=2e-10)
