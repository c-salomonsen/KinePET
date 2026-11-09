# SPDX-FileCopyrightText: 2026 Christian Salomonsen
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import kinepet

MODEL_TYPES = [(kinepet.Irreversible2TCM, "irr"), (kinepet.Reversible2TCM, "rev")]
BLOOD_FRACTIONS = [kinepet.FittedVB(), kinepet.FixedVB(0.0), kinepet.FixedVB(0.05)]


def protocol():
    time = np.array([0., .0833, .1667, .25, .3333, .5, .75, 1., 1.5, 2., 3., 5., 8., 12., 20., 30., 45.])
    aif = 12. * np.exp(-time / .3) + 1.5 * np.exp(-time / 6.)
    return time, aif


@pytest.mark.parametrize("model_type,name", MODEL_TYPES)
@pytest.mark.parametrize("vb", BLOOD_FRACTIONS)
@pytest.mark.parametrize("iterations", [0, 1, 200])
@pytest.mark.parametrize("aif_mode", ["plain", "fixed", "fitted"])
def test_model_objects_preserve_legacy_single_tac_results(model_type, name, vb, iterations, aif_mode):
    time, aif = protocol()
    model = model_type(vb=vb)
    fraction = vb.value if isinstance(vb, kinepet.FixedVB) else .04
    params = dict(K1=.09, k2=.22, k3=.11, k4=.06 if name == "rev" else 0., vB=fraction)
    tac = kinepet.evaluate_model(time, aif, model=name, **params)
    legacy_options = dict(fit_vb=isinstance(vb, kinepet.FittedVB), fixed_vb=fraction)
    aif_options = {} if aif_mode == "plain" else dict(
        fit_delay=aif_mode == "fitted", fixed_delay=.03,
        fit_dispersion=aif_mode == "fitted", fixed_dispersion=.02)
    legacy = kinepet.fit_one_tac(tac, time, aif, model=name, max_iterations=iterations,
                                 **legacy_options, **aif_options)
    typed = kinepet.fit_one_tac(tac, time, aif, model=model, max_iterations=iterations, **aif_options)
    assert asdict(typed) == asdict(legacy)
    assert kinepet.default_bounds(model, **aif_options) == kinepet.default_bounds(
        name, **legacy_options, **aif_options)
    assert_array_equal(kinepet.evaluate_model(time, aif, model=model, **params), tac)


@pytest.mark.parametrize("model_type,name", MODEL_TYPES)
@pytest.mark.parametrize("vb", BLOOD_FRACTIONS)
@pytest.mark.parametrize("per_tac", [False, True])
def test_model_objects_preserve_batch_shapes_and_results(model_type, name, vb, per_tac):
    time, aif = protocol()
    fraction = vb.value if isinstance(vb, kinepet.FixedVB) else .04
    model = model_type(vb=vb)
    time_columns = [time, time * 1.03] if per_tac else [time, time]
    aif_columns = [aif, aif * .85] if per_tac else [aif, aif]
    tacs = np.column_stack([
        kinepet.evaluate_model(t, cp, model=name, K1=.09, k2=.22, k3=.11,
                               k4=.06 if name == "rev" else 0., vB=fraction)
        for t, cp in zip(time_columns, aif_columns)
    ])
    fit_time = np.column_stack(time_columns) if per_tac else time
    fit_aif = np.column_stack(aif_columns) if per_tac else aif
    weights = np.linspace(.5, 1.5, time.size)
    if per_tac:
        weights = np.column_stack([weights, weights * .9])
    legacy = kinepet.fit_tacs(tacs, fit_time, fit_aif, model=name, weights=weights,
                              fit_vb=isinstance(vb, kinepet.FittedVB), fixed_vb=fraction, threads=1)
    typed = kinepet.fit_tacs(tacs, fit_time, fit_aif, model=model, weights=weights, threads=2)
    for key, value in legacy.as_dict().items():
        assert_array_equal(typed.as_dict()[key], value, err_msg=key)


@pytest.mark.parametrize("model_type,name", MODEL_TYPES)
def test_default_model_objects_fit_vb(model_type, name):
    model = model_type()
    assert isinstance(model.vb, kinepet.FittedVB)
    assert kinepet.default_bounds(model) == kinepet.default_bounds(name)
    with pytest.raises(FrozenInstanceError):
        model.vb = kinepet.FixedVB(0.)


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf")])
def test_fixed_blood_fraction_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite and between 0 and 1"):
        kinepet.FixedVB(value)


@pytest.mark.parametrize("model_type,name", MODEL_TYPES)
def test_model_objects_reject_ambiguous_configuration(model_type, name):
    time, aif = protocol()
    model = model_type(vb=kinepet.FixedVB(.05))
    with pytest.raises(TypeError, match="vb must be"):
        model_type(vb=.05)
    for options in ({"fit_vb": False}, {"fixed_vb": .05}):
        with pytest.raises(ValueError, match="Set vb on the model object"):
            kinepet.default_bounds(model, **options)
        with pytest.raises(ValueError, match="Set vb on the model object"):
            kinepet.fit_one_tac(aif, time, aif, model=model, **options)
        with pytest.raises(ValueError, match="Set vb on the model object"):
            kinepet.fit_tacs(aif[:, None], time, aif, model=model, **options)
    with pytest.raises(ValueError, match="conflicts"):
        kinepet.evaluate_model(time, aif, model=model, K1=.1, k2=.2, k3=.1, vB=.04)
    with pytest.raises(TypeError, match="model must be"):
        kinepet.default_bounds(object())


@pytest.mark.parametrize("model_type,name", MODEL_TYPES)
def test_fixed_model_supplies_vascular_fraction_for_evaluation(model_type, name):
    time, aif = protocol()
    params = dict(K1=.09, k2=.22, k3=.11, k4=.06 if name == "rev" else 0.)
    for fraction in (0., .05, 1.):
        model = model_type(vb=kinepet.FixedVB(fraction))
        assert_array_equal(
            kinepet.evaluate_model(time, aif, model=model, **params),
            kinepet.evaluate_model(time, aif, model=name, vB=fraction, **params))
    # The default FittedVB evaluation keeps the existing numeric default, .05.
    assert_array_equal(kinepet.evaluate_model(time, aif, model=model_type(), **params),
                       kinepet.evaluate_model(time, aif, model=name, **params))


@pytest.mark.parametrize("first_time", [0., .02])
def test_repeated_root_reversible_fallback_matches_one_tissue_limit(first_time):
    time = np.linspace(first_time, 30., 301)
    aif = 12. * np.exp(-time / .3) + 1.5 * np.exp(-time / 6.)
    # k3=0 makes b identically zero; k2=k4 selects the repeated-root fallback.
    common = dict(K1=.09, k2=.2, k3=0., vB=.04)
    rev = kinepet.evaluate_model(time, aif, model=kinepet.Reversible2TCM(), k4=.2, **common)
    irr = kinepet.evaluate_model(time, aif, model=kinepet.Irreversible2TCM(), **common)
    np.testing.assert_allclose(rev, irr, rtol=1e-8, atol=1e-10)
