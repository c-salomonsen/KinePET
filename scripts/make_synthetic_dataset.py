from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
        z = exp_conv_step(0.0, lam, time[0], 0.0, float(aif[0]))
        icp = 0.5 * float(aif[0]) * float(time[0])
    out[0] = (1 - vb) * (k1 / lam * (k3 * icp + k2 * z)) + vb * aif[0]
    for i in range(len(time) - 1):
        dt = float(time[i + 1] - time[i])
        z = exp_conv_step(z, lam, dt, float(aif[i]), float(aif[i + 1]))
        icp += 0.5 * float(aif[i] + aif[i + 1]) * dt
        ct = k1 / lam * (k3 * icp + k2 * z)
        out[i + 1] = (1 - vb) * ct + vb * aif[i + 1]
    return out


def forward_rev(
    time: np.ndarray,
    aif: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    vb: float,
) -> np.ndarray:
    rate_sum = k2 + k3 + k4
    disc = max(rate_sum**2 - 4.0 * k2 * k4, 0.0)
    root = np.sqrt(disc)
    a1 = 0.5 * (rate_sum + root)
    a2 = 0.5 * (rate_sum - root)
    denom = max(abs(a1 - a2), 1e-12)
    c1 = k1 * (a1 - k3 - k4) / denom
    c2 = -k1 * (a2 - k3 - k4) / denom

    z1 = 0.0
    z2 = 0.0
    out = np.zeros_like(time)
    if time[0] > 0:
        z1 = exp_conv_step(0.0, a1, float(time[0]), 0.0, float(aif[0]))
        z2 = exp_conv_step(0.0, a2, float(time[0]), 0.0, float(aif[0]))
    out[0] = (1 - vb) * (c1 * z1 + c2 * z2) + vb * aif[0]
    for i in range(len(time) - 1):
        dt = float(time[i + 1] - time[i])
        z1 = exp_conv_step(z1, a1, dt, float(aif[i]), float(aif[i + 1]))
        z2 = exp_conv_step(z2, a2, dt, float(aif[i]), float(aif[i + 1]))
        out[i + 1] = (1 - vb) * (c1 * z1 + c2 * z2) + vb * aif[i + 1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a small synthetic test dataset.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", choices=("rev", "irr"), default="rev")
    parser.add_argument("--noise-std", type=float, default=0.0)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    time = np.array(
        [0.0, 0.0833, 0.1667, 0.25, 0.3333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0],
        dtype=np.float64,
    )
    aif = 12.0 * np.exp(-time / 0.3) + 1.5 * np.exp(-time / 6.0)

    depth, height, width = 4, 4, 4
    image = np.zeros((time.size, depth, height, width), dtype=np.float64)

    truth = {
        "K1": np.zeros((depth, height, width), dtype=np.float64),
        "k2": np.zeros((depth, height, width), dtype=np.float64),
        "k3": np.zeros((depth, height, width), dtype=np.float64),
        "k4": np.zeros((depth, height, width), dtype=np.float64),
        "vB": np.zeros((depth, height, width), dtype=np.float64),
        "Ki": np.zeros((depth, height, width), dtype=np.float64),
    }

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                k1 = 0.08 + 0.03 * z / max(depth - 1, 1)
                k2 = 0.10 + 0.05 * y / max(height - 1, 1)
                k3 = 0.06 + 0.04 * x / max(width - 1, 1)
                k4 = 0.03 + 0.03 * (z + y + x) / max(depth + height + width - 3, 1)
                vb = 0.04 + 0.02 * (z % 2)

                if args.model == "rev":
                    tac = forward_rev(time, aif, k1, k2, k3, k4, vb)
                else:
                    k4 = 0.0
                    tac = forward_irrev(time, aif, k1, k2, k3, vb)

                if args.noise_std > 0:
                    tac = tac + rng.normal(0.0, args.noise_std, size=tac.shape)

                image[:, z, y, x] = tac
                truth["K1"][z, y, x] = k1
                truth["k2"][z, y, x] = k2
                truth["k3"][z, y, x] = k3
                truth["k4"][z, y, x] = k4
                truth["vB"][z, y, x] = vb
                truth["Ki"][z, y, x] = k1 * k3 / (k2 + k3)

    np.save(args.output_dir / "image.npy", image)
    np.save(args.output_dir / "time.npy", time)
    np.save(args.output_dir / "aif.npy", aif)
    for key, value in truth.items():
        if key == "k4" and args.model == "irr":
            continue
        np.save(args.output_dir / f"truth_{key}.npy", value)


if __name__ == "__main__":
    main()
