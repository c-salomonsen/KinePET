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


def exp_conv_step(z: float, rate: float, dt: float, c0: float, c1: float) -> float:
    if dt <= 0:
        return z
    if abs(rate) < 1e-12:
        return z + 0.5 * (c0 + c1) * dt
    decay = np.exp(-rate * dt)
    i0 = (1.0 - decay) / rate
    i1 = (1.0 - decay * (1.0 + rate * dt)) / (rate**2)
    slope = (c1 - c0) / dt
    return decay * z + c1 * i0 - slope * i1


def forward_irrev(time: np.ndarray, aif: np.ndarray, k1: float, k2: float, k3: float, vb: float) -> np.ndarray:
    lam = max(k2 + k3, 1e-12)
    z = 0.0
    icp = 0.0
    out = np.zeros_like(time)
    if time[0] > 0:
        z = exp_conv_step(0.0, lam, float(time[0]), 0.0, float(aif[0]))
        icp = 0.5 * float(aif[0]) * float(time[0])
    out[0] = (1 - vb) * (k1 / lam * (k3 * icp + k2 * z)) + vb * aif[0]
    for i in range(len(time) - 1):
        dt = float(time[i + 1] - time[i])
        z = exp_conv_step(z, lam, dt, float(aif[i]), float(aif[i + 1]))
        icp += 0.5 * float(aif[i] + aif[i + 1]) * dt
        out[i + 1] = (1 - vb) * (k1 / lam * (k3 * icp + k2 * z)) + vb * aif[i + 1]
    return out


def make_tac_batch_dataset(output_dir: Path) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    time = np.array(
        [0.0, 0.0833, 0.1667, 0.25, 0.3333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0],
        dtype=np.float64,
    )
    aif = np.stack(
        [
            12.0 * np.exp(-time / 0.3) + 1.5 * np.exp(-time / 6.0),
            8.0 * np.exp(-time / 0.45) + 2.0 * np.exp(-time / 8.0),
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
            forward_irrev(time, aif[:, 1], truth["K1"][1], truth["k2"][1], truth["k3"][1], truth["vB"][1]),
        ],
        axis=1,
    )
    np.save(output_dir / "time.npy", time)
    np.save(output_dir / "aif.npy", aif)
    np.save(output_dir / "tacs.npy", tacs)
    return truth


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
    shutil.rmtree(ROOT / "data" / "synthetic_rev_rescue_out", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_rev_via_tacs_out", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_irr", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_irr_out", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_tac_batch", ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "synthetic_tac_batch_out", ignore_errors=True)

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
    run(
        [
            str(BUILD / "tcm_fit"),
            "--input-dir",
            "data/synthetic_rev",
            "--output-dir",
            "data/synthetic_rev_rescue_out",
            "--model",
            "rev",
            "--mask-threshold",
            "0.0",
            "--rescue-refit",
            "1",
        ],
        cwd=ROOT,
    )
    run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "scripts/fit_image_via_tacs.py",
            "--input-dir",
            "data/synthetic_rev",
            "--output-dir",
            "data/synthetic_rev_via_tacs_out",
            "--tac-fit",
            str(BUILD / "tcm_fit_tacs"),
            "--model",
            "rev",
            "--mask-threshold",
            "0.0",
        ],
        cwd=ROOT,
    )
    tac_truth = make_tac_batch_dataset(ROOT / "data" / "synthetic_tac_batch")
    run(
        [
            str(BUILD / "tcm_fit_tacs"),
            "--input-dir",
            "data/synthetic_tac_batch",
            "--output-dir",
            "data/synthetic_tac_batch_out",
            "--model",
            "irr",
        ],
        cwd=ROOT,
    )

    check_mae("rev", "K1", 5e-3)
    check_mae("rev", "k4", 5e-3)
    check_mae("irr", "K1", 5e-3)
    check_mae("irr", "vB", 5e-3)
    check_map_mae(
        ROOT / "data" / "synthetic_rev_via_tacs_out",
        ROOT / "data" / "synthetic_rev",
        "K1",
        5e-3,
    )
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
    rescue_out = ROOT / "data" / "synthetic_rev_rescue_out"
    for name in ["rescue_target.npy", "rescue_applied.npy", "rescue_unresolved.npy"]:
        if not (rescue_out / name).exists():
            raise SystemExit(f"Smoke test failed: {name} was not written for rescue run")
    if not np.isfinite(np.load(rescue_out / "Ki.npy")).all():
        raise SystemExit("Smoke test failed: rescue run Ki output contains non-finite values")
    tac_out = ROOT / "data" / "synthetic_tac_batch_out"
    for key, truth in tac_truth.items():
        pred = np.load(tac_out / f"{key}.npy")
        mae = np.nanmean(np.abs(pred - truth))
        print(f"synthetic_tac_batch {key} mean absolute error: {mae:.6f}")
        if mae > 5e-3:
            raise SystemExit(f"Smoke test failed for synthetic_tac_batch {key}: error too high")


if __name__ == "__main__":
    main()
