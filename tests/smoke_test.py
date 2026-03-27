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


def main() -> None:
    shutil.rmtree(BUILD, ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev_out", ignore_errors=True)
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

    check_mae("rev", "K1", 5e-3)
    check_mae("rev", "k4", 5e-3)
    check_mae("irr", "K1", 5e-3)
    check_mae("irr", "vB", 5e-3)


if __name__ == "__main__":
    main()
