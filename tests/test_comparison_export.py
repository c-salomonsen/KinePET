"""Exercise the comparison CSV without private data or reference packages."""
import csv
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

import kinepet

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_voi_reference_fits.py"
spec = importlib.util.spec_from_file_location("comparison_export", SCRIPT)
export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export
spec.loader.exec_module(export)


@pytest.mark.parametrize("unit", ["minutes", "seconds"])
def test_synthetic_export(tmp_path, unit):
    time = np.array([0, .1, .3, .5, 1, 2, 5, 10, 20, 40, 80, 100.])
    aif = 10 * np.exp(-time) + np.exp(-time / 20)
    tac = kinepet.evaluate_model(time, aif, model="rev", K1=.1, k2=.2, k3=.1, k4=.04)
    saved_time = time * 60 if unit == "seconds" else time
    np.savez(tmp_path / "sample.npz", time=saved_time, aif=aif)
    with (tmp_path / "VOI_sample.pkl").open("wb") as handle:
        pickle.dump({"VOI_t": saved_time, "VOI_A": {"Brain": tac, "Myocardium": tac}}, handle)
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(SCRIPT), "--raw-npz-glob", str(tmp_path / "*.npz"),
                    "--voi-glob", str(tmp_path / "*.pkl"), "--npz-time-unit", unit,
                    "--voi-time-unit", unit, "--output-dir", str(out)], check=True, capture_output=True)
    with (out / "voi_reference_fits.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    for row in rows:
        assert int(row["status"]) in (1, 2, 3, 4)
        np.testing.assert_allclose(float(row["common_rmse_native_grid"]), float(row["error_value"]), atol=1e-14)
    with np.load(out / "voi_reference_curves.npz") as arrays:
        np.testing.assert_allclose(arrays["time_minutes"][0], time)
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["methods"] == ["cpp"]
    assert metadata["pypet_module"] is None
    assert metadata["cpp_fit_delay"] is False


def test_time_units_are_explicit():
    time = np.array([0., 90., 120.])
    np.testing.assert_array_equal(export.to_minutes(time, "minutes"), time)
    np.testing.assert_array_equal(export.to_minutes(time, "seconds"), [0., 1.5, 2.])
    with pytest.raises(ValueError):
        export.to_minutes(time, "guess")


def test_common_metric_uses_parameters_not_reference_error():
    time = np.arange(6.)
    aif = np.ones(6)
    params = dict(K1=.1, k2=.2, k3=.1, k4=.04, vB=.05)
    tac = kinepet.evaluate_model(time, aif, model="rev", **params)
    sample = export.SampleCurves("x", Path("x.npz"), Path("x.pkl"), None, None, time, aif, tac, tac)
    rows = [dict(params, model="rev", region="Brain", error_value=123., method=method)
            for method in ("cpp", "pypet", "lls")]
    export.add_common_rmse(rows, sample)
    assert all(row["common_rmse_native_grid"] == 0 for row in rows)
    assert all(row["error_value"] == 123 for row in rows)
