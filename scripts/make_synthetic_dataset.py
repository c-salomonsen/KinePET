# SPDX-FileCopyrightText: 2026 Christian Salomonsen
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kinepet import evaluate_model


def make_aif(time: np.ndarray) -> np.ndarray:
    return 12.0 * np.exp(-time / 0.3) + 1.5 * np.exp(-time / 6.0)


def sample_piecewise_linear_with_derivative(
    time: np.ndarray,
    signal: np.ndarray,
    t: float,
) -> tuple[float, float]:
    if t < 0.0:
        return 0.0, 0.0
    if time.size == 1:
        if time[0] > 0.0 and t < float(time[0]):
            return float(signal[0]) * (t / float(time[0])), float(signal[0]) / float(time[0])
        return float(signal[0]), 0.0
    if t < float(time[0]):
        if float(time[0]) <= 0.0:
            return float(signal[0]), 0.0
        return float(signal[0]) * (t / float(time[0])), float(signal[0]) / float(time[0])
    if t > float(time[-1]):
        return float(signal[-1]), 0.0

    upper = int(np.searchsorted(time, t, side="right"))
    idx = max(0, upper - 1)
    if idx + 1 >= time.size:
        return float(signal[-1]), 0.0
    t0 = float(time[idx])
    t1 = float(time[idx + 1])
    if t1 <= t0:
        return float(signal[idx]), 0.0

    slope = (float(signal[idx + 1]) - float(signal[idx])) / (t1 - t0)
    return float(signal[idx]) + (t - t0) * slope, slope


def transform_aif(time: np.ndarray, aif: np.ndarray, delay: float, dispersion: float) -> np.ndarray:
    tau = max(float(dispersion), 0.0)
    out = np.zeros_like(aif)
    for i, t in enumerate(time):
        value, deriv = sample_piecewise_linear_with_derivative(time, aif, float(t) + float(delay))
        out[i] = max(0.0, value + tau * deriv)
    return out


def make_parameters(num_tacs: int, model: str) -> dict[str, np.ndarray]:
    index = np.arange(num_tacs, dtype=np.float64)
    scale = np.maximum(num_tacs - 1, 1)

    params = {
        "K1": 0.08 + 0.04 * index / scale,
        "k2": 0.10 + 0.06 * ((index * 3) % max(num_tacs, 2)) / scale,
        "k3": 0.05 + 0.05 * ((index * 5) % max(num_tacs, 2)) / scale,
        "vB": 0.03 + 0.03 * (index % 4) / 3.0,
        "delay": np.zeros(num_tacs, dtype=np.float64),
        "dispersion": np.zeros(num_tacs, dtype=np.float64),
    }
    if model == "rev":
        params["k4"] = 0.02 + 0.04 * ((index * 7) % max(num_tacs, 2)) / scale
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic TAC batch dataset.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", choices=("rev", "irr"), default="rev")
    parser.add_argument("--num-tacs", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--dispersion", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_tacs <= 0:
        raise SystemExit("--num-tacs must be positive")

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    time = np.array(
        [0.0, 0.0833, 0.1667, 0.25, 0.3333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0],
        dtype=np.float64,
    )
    aif = make_aif(time)
    effective_aif = (
        transform_aif(time, aif, args.delay, args.dispersion)
        if args.delay != 0.0 or args.dispersion != 0.0
        else aif
    )
    params = make_parameters(args.num_tacs, args.model)
    params["delay"].fill(args.delay)
    params["dispersion"].fill(args.dispersion)

    # Share the compiled kinetics and AIF interpolation with the fitter.
    n_tacs = len(params["K1"])
    tacs = np.empty((time.size, n_tacs), dtype=np.float64)
    for i in range(n_tacs):
        tacs[:, i] = evaluate_model(
            time,
            effective_aif,
            model=args.model,
            K1=float(params["K1"][i]),
            k2=float(params["k2"][i]),
            k3=float(params["k3"][i]),
            k4=float(params["k4"][i]) if args.model == "rev" else 0.0,
            vB=float(params["vB"][i]),
        )

    if args.noise_std > 0.0:
        tacs = tacs + rng.normal(0.0, args.noise_std, size=tacs.shape)

    np.save(args.output_dir / "tacs.npy", np.asarray(tacs, dtype=np.float64))
    np.save(args.output_dir / "time.npy", time)
    np.save(args.output_dir / "aif.npy", aif)

    ki = params["K1"] * params["k3"] / np.maximum(params["k2"] + params["k3"], 1e-12)
    truth = dict(params)
    truth["Ki"] = ki
    for name, values in truth.items():
        if name == "k4" and args.model == "irr":
            continue
        np.save(args.output_dir / f"truth_{name}.npy", np.asarray(values, dtype=np.float64))


if __name__ == "__main__":
    main()
