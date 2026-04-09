from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def check_mae(model: str, key: str, threshold: float) -> None:
    data_dir = ROOT / "data" / f"synthetic_{model}"
    out_dir = ROOT / "data" / f"synthetic_{model}_out"
    pred = np.load(out_dir / f"{key}.npy")
    truth = np.load(data_dir / f"truth_{key}.npy")
    mae = np.nanmean(np.abs(pred - truth))
    print(f"{model} {key} mean absolute error: {mae:.6f}")
    if mae > threshold:
        raise SystemExit(f"Smoke test failed for {model} {key}: error too high")


def check_map_mae(output_dir: Path, truth_dir: Path, key: str, threshold: float) -> None:
    pred = np.load(output_dir / f"{key}.npy")
    truth = np.load(truth_dir / f"truth_{key}.npy")
    mae = np.nanmean(np.abs(pred - truth))
    print(f"{output_dir.name} {key} mean absolute error: {mae:.6f}")
    if mae > threshold:
        raise SystemExit(f"Smoke test failed for {output_dir.name} {key}: error too high")


def main() -> None:
    shutil.rmtree(BUILD, ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev_out", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev_delay", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev_delay_out", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev_dispersion", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev_dispersion_out", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev_delaydisp", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev_delaydisp_out", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_irr", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_irr_out", ignore_errors=True)

    run(
        ["python3", "scripts/make_synthetic_dataset.py", "data/synthetic_rev", "--model", "rev"],
        cwd=ROOT,
    )
    run(
        ["python3", "scripts/make_synthetic_dataset.py", "data/synthetic_irr", "--model", "irr"],
        cwd=ROOT,
    )
    run(
        [
            "python3",
            "scripts/make_synthetic_dataset.py",
            "data/synthetic_rev_delay",
            "--model",
            "rev",
            "--delay",
            "0.08",
        ],
        cwd=ROOT,
    )
    run(
        [
            "python3",
            "scripts/make_synthetic_dataset.py",
            "data/synthetic_rev_dispersion",
            "--model",
            "rev",
            "--dispersion",
            "0.05",
        ],
        cwd=ROOT,
    )
    run(
        [
            "python3",
            "scripts/make_synthetic_dataset.py",
            "data/synthetic_rev_delaydisp",
            "--model",
            "rev",
            "--delay",
            "0.03",
            "--dispersion",
            "0.02",
        ],
        cwd=ROOT,
    )
    run(["cmake", "-S", ".", "-B", "build"], cwd=ROOT)
    run(["cmake", "--build", "build"], cwd=ROOT)
    run(
        [
            str(BUILD / "tcm_fit"),
            "--input-dir",
            "data/synthetic_rev",
            "--output-dir",
            "data/synthetic_rev_out",
            "--model",
            "rev",
            "--mask-threshold",
            "0.0",
        ],
        cwd=ROOT,
    )
    run(
        [
            str(BUILD / "tcm_fit"),
            "--input-dir",
            "data/synthetic_irr",
            "--output-dir",
            "data/synthetic_irr_out",
            "--model",
            "irr",
            "--mask-threshold",
            "0.0",
        ],
        cwd=ROOT,
    )
    run(
        [
            str(BUILD / "tcm_fit"),
            "--input-dir",
            "data/synthetic_rev_delay",
            "--output-dir",
            "data/synthetic_rev_delay_out",
            "--model",
            "rev",
            "--fit-delay",
            "1",
            "--mask-threshold",
            "0.0",
        ],
        cwd=ROOT,
    )
    run(
        [
            str(BUILD / "tcm_fit"),
            "--input-dir",
            "data/synthetic_rev_dispersion",
            "--output-dir",
            "data/synthetic_rev_dispersion_out",
            "--model",
            "rev",
            "--fit-dispersion",
            "1",
            "--mask-threshold",
            "0.0",
        ],
        cwd=ROOT,
    )
    run(
        [
            str(BUILD / "tcm_fit"),
            "--input-dir",
            "data/synthetic_rev_delaydisp",
            "--output-dir",
            "data/synthetic_rev_delaydisp_out",
            "--model",
            "rev",
            "--fit-delay",
            "1",
            "--fit-dispersion",
            "1",
            "--mask-threshold",
            "0.0",
        ],
        cwd=ROOT,
    )

    check_mae("rev", "K1", 5e-3)
    check_mae("rev", "k4", 5e-3)
    check_mae("irr", "K1", 5e-3)
    check_mae("irr", "vB", 5e-3)
    check_map_mae(
        ROOT / "data" / "synthetic_rev_delay_out",
        ROOT / "data" / "synthetic_rev_delay",
        "delay",
        5e-3,
    )
    check_map_mae(
        ROOT / "data" / "synthetic_rev_dispersion_out",
        ROOT / "data" / "synthetic_rev_dispersion",
        "dispersion",
        5e-3,
    )
    delay_disp_out = ROOT / "data" / "synthetic_rev_delaydisp_out"
    if not (delay_disp_out / "delay.npy").exists() or not (delay_disp_out / "dispersion.npy").exists():
        raise SystemExit("Smoke test failed: simultaneous delay/dispersion outputs were not written")
    if not np.isfinite(np.load(delay_disp_out / "delay.npy")).all():
        raise SystemExit("Smoke test failed: simultaneous delay output contains non-finite values")
    if not np.isfinite(np.load(delay_disp_out / "dispersion.npy")).all():
        raise SystemExit("Smoke test failed: simultaneous dispersion output contains non-finite values")


if __name__ == "__main__":
    main()
