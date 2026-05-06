from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.kinetics import Irreversible2TCMVB, Reversible2TCMVB


TIME_PROTOCOL = ((1, 30), (24, 5), (9, 20), (8, 300))
DT_S = 1.0
TINJ_S = 0.5


def feng_aif(t, tau, A1, A2, A3, L1, L2, L3):
    cp = (
        (A1 * (t - tau) - A2 - A3) * np.exp(L1 * (t - tau))
        + A2 * np.exp(L2 * (t - tau))
        + A3 * np.exp(L3 * (t - tau))
    )
    cp[: np.sum(t < tau)] = 0
    return cp


def frame_average(y: np.ndarray, t_s: np.ndarray, frame_durations_s: np.ndarray) -> np.ndarray:
    edges = np.concatenate([[0.0], np.cumsum(frame_durations_s)])
    out = np.empty(len(frame_durations_s), dtype=np.float64)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (t_s >= a) & (t_s < b)
        if np.any(m):
            out[i] = np.trapezoid(y[m], t_s[m]) / (b - a)
        else:
            out[i] = np.interp(0.5 * (a + b), t_s, y)
    return out


def generate_aif(
    t_fine_min: np.ndarray,
    t_fine_s: np.ndarray,
    frame_durations_s: np.ndarray,
    dt_s: float,
    *,
    tinj_s: float,
    tau: float,
    A1: float,
    r2: float,
    r3: float,
    L1: float,
    L2: float,
    L3: float,
) -> np.ndarray:
    k = max(1, int(round(tinj_s / dt_s)))
    kernel = np.ones(k, dtype=np.float64) / k
    A2, A3 = r2 * A1, r3 * A1

    cp = feng_aif(t_fine_min, tau, A1, A2, A3, L1, L2, L3)
    cp_inj = np.convolve(cp, kernel, mode="same")
    return frame_average(cp_inj, t_fine_s, frame_durations_s)


def generate_time(frame_protocol: tuple[tuple[int, int], ...], dt_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_durations_s = np.concatenate([np.full(n_frames, frame_dur, dtype=np.float64) for n_frames, frame_dur in frame_protocol])
    edges_s = np.concatenate([[0.0], np.cumsum(frame_durations_s)])
    mids_s = 0.5 * (edges_s[:-1] + edges_s[1:])

    t_fine_s = np.arange(0.0, edges_s[-1] + dt_s, dt_s, dtype=np.float64)
    t_fine_min = t_fine_s / 60.0
    time_min = mids_s / 60.0
    return t_fine_min, t_fine_s, time_min


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert simulated CSV parameter tables into TAC batch directories "
            "for the tcm_fit_tacs binary."
        )
    )
    parser.add_argument("--csv-glob", default="data/simulated-csv/*.csv")
    parser.add_argument("--output-root", type=Path, default=Path("data/simulated-tac-batches"))
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument("--tinj-s", type=float, default=TINJ_S)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_csvs(pattern: str) -> list[Path]:
    import glob

    return sorted(Path(path) for path in glob.glob(pattern))


def convert_csv(csv_path: Path, output_root: Path, dt_s: float, tinj_s: float, overwrite: bool) -> Path:
    df = pd.read_csv(csv_path, sep=";")
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    model = "rev" if "k4" in df.columns else "irr"
    compartment_model = Reversible2TCMVB() if model == "rev" else Irreversible2TCMVB()

    t_fine_min, t_fine_s, time_min = generate_time(TIME_PROTOCOL, dt_s)
    frame_durations_s = np.concatenate(
        [np.full(n_frames, frame_dur, dtype=np.float64) for n_frames, frame_dur in TIME_PROTOCOL]
    )

    n = len(df)
    T = len(time_min)
    aif = np.empty((T, n), dtype=np.float64)
    tacs = np.empty((T, n), dtype=np.float64)

    for j, row in enumerate(df.itertuples(index=False)):
        A1m = float(row.A1m)
        A2m = float(row.A2m)
        A3m = float(row.A3m)
        curve_aif = generate_aif(
            t_fine_min,
            t_fine_s,
            frame_durations_s,
            dt_s,
            tinj_s=tinj_s,
            tau=float(row.TAUm),
            A1=A1m,
            r2=A2m / A1m,
            r3=A3m / A1m,
            L1=float(row.L1m),
            L2=float(row.L2m),
            L3=float(row.L3m),
        )
        aif[:, j] = curve_aif

        if model == "rev":
            tacs[:, j] = compartment_model(
                curve_aif,
                time_min,
                K1=np.array([float(row.K1)]),
                k2=np.array([float(row.k2)]),
                k3=np.array([float(row.k3)]),
                k4=np.array([float(row.k4)]),
                vB=np.array([float(row.vB)]),
            ).reshape(-1)
        else:
            tacs[:, j] = compartment_model(
                curve_aif,
                time_min,
                K1=np.array([float(row.K1)]),
                k2=np.array([float(row.k2)]),
                k3=np.array([float(row.k3)]),
                vB=np.array([float(row.vB)]),
            ).reshape(-1)

    out_dir = output_root / csv_path.stem
    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "tacs.npy", tacs)
    np.save(out_dir / "aif.npy", aif)
    np.save(out_dir / "time.npy", time_min)
    np.save(out_dir / "sample_ids.npy", df["sample"].astype(str).to_numpy())
    df.to_csv(out_dir / "truth.csv", index=False)

    metadata = {
        "source_csv": str(csv_path.resolve()),
        "model": model,
        "num_tacs": n,
        "num_timepoints": T,
        "time_protocol": [list(pair) for pair in TIME_PROTOCOL],
        "dt_s": dt_s,
        "tinj_s": tinj_s,
        "aif_mode": "per_tac",
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return out_dir


def main() -> None:
    args = parse_args()
    csv_paths = discover_csvs(args.csv_glob)
    if not csv_paths:
        raise SystemExit(f"No CSV files matched: {args.csv_glob}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_paths:
        out_dir = convert_csv(csv_path, args.output_root, args.dt_s, args.tinj_s, args.overwrite)
        print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
