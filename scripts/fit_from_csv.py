# SPDX-FileCopyrightText: 2026 Christian Salomonsen
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal
from time import perf_counter
import logging

import numpy as np
import pandas as pd

from kinepet import evaluate_model, fit_tacs


ModelName = Literal["irr", "rev"]


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


def generate_curves(df: pd.DataFrame, dt_s: float, tinj_s: float, model: ModelName) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic TAC batch from a parameter table.

    Curves are generated with ``kinepet.evaluate_model`` so simulation and
    fitting use the same compiled kinetics and AIF interpolation conventions.
    """
    t_fine_min, t_fine_s, time_min = generate_time(TIME_PROTOCOL, dt_s)
    frame_durations_s = np.concatenate(
        [np.full(n_frames, frame_dur, dtype=np.float64) for n_frames, frame_dur in TIME_PROTOCOL]
    )

    n = len(df)
    T = len(time_min)
    aif = np.empty((T, n), dtype=np.float64)
    tacs = np.empty((T, n), dtype=np.float64)
    time_min = np.ascontiguousarray(time_min)

    for j, row in enumerate(df.itertuples(index=False)):
        A1m = float(row.A1m)
        curve_aif = generate_aif(
            t_fine_min,
            t_fine_s,
            frame_durations_s,
            dt_s,
            tinj_s=tinj_s,
            tau=float(row.TAUm),
            A1=A1m,
            r2=float(row.A2m) / A1m,
            r3=float(row.A3m) / A1m,
            L1=float(row.L1m),
            L2=float(row.L2m),
            L3=float(row.L3m),
        )
        aif[:, j] = curve_aif
        tacs[:, j] = evaluate_model(
            time_min,
            np.ascontiguousarray(curve_aif),
            model=model,
            K1=float(row.K1),
            k2=float(row.k2),
            k3=float(row.k3),
            k4=float(row.k4) if model == "rev" else 0.0,
            vB=float(row.vB),
        )

    time_out = np.repeat(time_min[:, None], aif.shape[1], axis=1)
    return tacs, aif, time_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TACs from CSV and fit them")
    parser.add_argument("INPUT_FILE", type=Path, help="Path to the input CSV file")
    parser.add_argument("OUTPUT_FILE", type=Path, help="Path to the output CSV file")
    parser.add_argument("--model", choices=["irr", "rev"], default="rev", help="Compartment model to use for fitting")
    parser.add_argument("--fit-vb", action="store_true", help="Whether to fit vB or keep it fixed")
    parser.add_argument("--fixed-vb", type=float, default=0.05, help="Value to use for fixed vB (ignored if --fit-vb is set)")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads to use for fitting (default: all available)")
    parser.add_argument("--max-iterations", type=int, default=40, help="Maximum number of iterations for fitting")
    parser.add_argument("--csv-sep", type=str, default=";", help="Separator used in the input CSV file (default: ';')")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity")
    args = parser.parse_args()

    # Logger setup
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(loglevel)
    console_handler.setLevel(loglevel)
    logger.addHandler(console_handler)

    df = pd.read_csv(args.INPUT_FILE, sep=args.csv_sep)
    logger.debug(f"Read csv: {args.INPUT_FILE}, rows: {len(df)}")

    model: ModelName = args.model

    tacs, aifs, times = generate_curves(df, dt_s=DT_S, tinj_s=TINJ_S, model=model)

    logger.debug(f"Generated curves. {tacs.shape = }, {aifs.shape = }, {times.shape = }")

    start = perf_counter()
    fit = fit_tacs(
        tacs,
        times,
        aifs,
        model=model,
        fit_vb=args.fit_vb,
        fixed_vb=args.fixed_vb,
        threads=args.threads,
        max_iterations=args.max_iterations,
    )
    duration = perf_counter() - start

    logger.info(f"Completed {fit.fitted_tacs} / {len(fit.status)} successful fits. Took {duration:.6f} s")

    out_df = df.copy()
    out_df["K1_fit"] = fit.K1
    out_df["k2_fit"] = fit.k2
    out_df["k3_fit"] = fit.k3
    if model == "rev":
        out_df["k4_fit"] = fit.k4
    out_df["vB_fit"] = fit.vB
    out_df["Ki_fit"] = fit.Ki
    out_df["rmse_fit"] = fit.rmse
    out_df["iterations_fit"] = fit.iterations
    out_df["status_fit"] = fit.status

    out_df.to_csv(args.OUTPUT_FILE, index=False, sep=args.csv_sep)
    logger.debug(f"Saved results to csv: {args.OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
