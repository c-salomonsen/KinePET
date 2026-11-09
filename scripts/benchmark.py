# SPDX-FileCopyrightText: 2026 Christian Salomonsen
# SPDX-License-Identifier: AGPL-3.0-only

"""Reproducible synthetic throughput/accuracy check; no patient data required."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time as clock

import numpy as np
import kinepet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tacs", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("output/benchmark.json"))
    args = parser.parse_args()
    if args.num_tacs < 1 or args.repeats < 1 or args.threads < 0:
        parser.error("num-tacs/repeats must be positive and threads nonnegative")
    rng = np.random.default_rng(args.seed)
    time = np.array([0., .0833, .1667, .25, .3333, .5, .75, 1., 1.5, 2., 3., 5., 8., 12., 20., 30., 45.])
    aif = 12 * np.exp(-time / .3) + 1.5 * np.exp(-time / 6)
    ranges = dict(K1=(.05, .2), k2=(.1, .3), k3=(.05, .15), k4=(.02, .08), vB=(.02, .08))
    truth = {key: rng.uniform(*limits, args.num_tacs) for key, limits in ranges.items()}
    report = {
        "kinepet": importlib.metadata.version("kinepet"), "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "seed": args.seed, "threads_requested": args.threads, "num_tacs": args.num_tacs,
        "repeats": args.repeats, "noise": "none", "time_minutes": time.tolist(),
        "aif": aif.tolist(), "parameter_ranges": ranges,
        "fit_settings": dict(fit_vb=True, fit_delay=False, fit_dispersion=False,
                             max_iterations=200, gradient_tolerance=1e-10, cost_tolerance=1e-16),
        "models": {},
        "scope": "Synthetic consistency and local throughput; not a reproduction of paper timings or an independent physical validation.",
    }
    root = Path(__file__).resolve().parents[1]
    try:
        report["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        report["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        report["git_revision"] = None
    for model in ("irr", "rev"):
        names = ("K1", "k2", "k3", "vB") + (("k4",) if model == "rev" else ())
        tacs = np.column_stack([
            kinepet.evaluate_model(time, aif, model=model, **{key: truth[key][i] for key in names})
            for i in range(args.num_tacs)
        ])
        # Exercise initialization/OpenMP before timed runs.
        kinepet.fit_tacs(tacs[:, :1], time, aif, model=model, threads=args.threads, **report["fit_settings"])
        durations = []
        for _ in range(args.repeats):
            start = clock.perf_counter()
            fit = kinepet.fit_tacs(tacs, time, aif, model=model, threads=args.threads, **report["fit_settings"])
            durations.append(clock.perf_counter() - start)
        finite = np.isfinite(fit.rmse)
        report["models"][model] = {
            "seconds": durations, "median_tacs_per_second": args.num_tacs / float(np.median(durations)),
            "status_counts": {str(code): int(np.sum(fit.status_code == code)) for code in range(6)},
            "failure_fraction": float(np.mean(np.isin(fit.status_code, [0, 5]))),
            "max_iterations_fraction": float(np.mean(fit.status_code == 4)),
            "median_rmse_native_grid": float(np.median(fit.rmse[finite])) if finite.any() else None,
            "max_rmse_native_grid": float(np.max(fit.rmse[finite])) if finite.any() else None,
            "median_relative_parameter_error": {
                key: float(np.median(np.abs(getattr(fit, key)[finite] - truth[key][finite]) / truth[key][finite]))
                if finite.any() else None for key in names
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
