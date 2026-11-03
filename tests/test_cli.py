"""CLI/API parity checks; run after building tcm_fit_tacs with CMake."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import kinepet

_DEFAULT_CLI = Path(__file__).resolve().parents[1] / "build" / (
    "tcm_fit_tacs.exe" if os.name == "nt" else "tcm_fit_tacs")
CLI = Path(os.environ.get("KINEPET_CLI", _DEFAULT_CLI)).resolve()
pytestmark = pytest.mark.skipif(not CLI.is_file(), reason="build tcm_fit_tacs or set KINEPET_CLI")


@pytest.mark.parametrize("name", ["irr", "rev"])
@pytest.mark.parametrize("vb", [None, 0., .05])
@pytest.mark.parametrize("per_tac", [False, True])
@pytest.mark.parametrize("aif_mode", ["plain", "fixed", "fitted"])
def test_cli_matches_python_api(tmp_path, name, vb, per_tac, aif_mode):
    time = np.array([0., .0833, .1667, .25, .3333, .5, .75, 1., 1.5, 2., 3., 5., 8., 12., 20., 30., 45.])
    aif = 12. * np.exp(-time / .3) + 1.5 * np.exp(-time / 6.)
    fraction = .04 if vb is None else vb
    times = [time, time * 1.03] if per_tac else [time, time]
    aifs = [aif, aif * .85] if per_tac else [aif, aif]
    tacs = np.column_stack([
        kinepet.evaluate_model(t, cp, model=name, K1=.09, k2=.22, k3=.11,
                               k4=.06 if name == "rev" else 0., vB=fraction)
        for t, cp in zip(times, aifs)
    ])
    fit_time = np.column_stack(times) if per_tac else time
    fit_aif = np.column_stack(aifs) if per_tac else aif[:, None]
    weights = np.linspace(.5, 1.5, len(time))
    if per_tac:
        weights = np.column_stack([weights, weights * .9])
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for key, value in {"time": fit_time, "aif": fit_aif, "tacs": tacs, "weights": weights}.items():
        np.save(input_dir / f"{key}.npy", value)
    output_dir = tmp_path / "output"
    fitted_aif = aif_mode == "fitted"
    aif_options = {} if aif_mode == "plain" else dict(
        fit_delay=fitted_aif, fixed_delay=.03,
        fit_dispersion=fitted_aif, fixed_dispersion=.02)
    args = [str(CLI), "--input-dir", str(input_dir), "--output-dir", str(output_dir),
            "--model", name, "--fixed-vb", str(fraction), "--fit-vb", str(int(vb is None)),
            "--threads", "2", "--max-iter", "100", "--weights-file", str(input_dir / "weights.npy")]
    if aif_mode != "plain":
        args += ["--fixed-delay", ".03", "--fit-delay", str(int(fitted_aif)),
                 "--fixed-dispersion", ".02", "--fit-dispersion", str(int(fitted_aif))]
    subprocess.run(args, check=True, capture_output=True, text=True)
    model_type = kinepet.Reversible2TCM if name == "rev" else kinepet.Irreversible2TCM
    model = model_type(vb=kinepet.FittedVB() if vb is None else kinepet.FixedVB(vb))
    expected = kinepet.fit_tacs(tacs, fit_time, fit_aif, weights=weights, model=model,
                                threads=1, max_iterations=100, **aif_options)
    names = {"K1", "k2", "k3", "vB", "Ki", "rmse"}
    if name == "rev":
        names.add("k4")
    if aif_mode != "plain":
        names.update(["delay", "dispersion"])
    for key in names:
        values = np.load(output_dir / f"{key}.npy")
        assert values.shape == (2,)
        assert_allclose(values, getattr(expected, key), rtol=1e-10, atol=1e-12, err_msg=key)
    assert_array_equal(np.load(output_dir / "status.npy"), expected.status_code)
    assert_array_equal(np.load(output_dir / "iterations.npy"), expected.iterations)
    assert {p.name for p in output_dir.iterdir()} == {
        *(f"{name}.npy" for name in names), "status.npy", "iterations.npy", "run.txt"}
    manifest = (output_dir / "run.txt").read_text()
    assert f"model=2tcm-{'reversible' if name == 'rev' else 'irreversible'}\n" in manifest
    assert f"shared_aif={int(not per_tac)}\n" in manifest
