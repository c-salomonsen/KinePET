from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

CELL_COUNTER = 0


def lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines(keepends=True)


def next_cell_id() -> str:
    global CELL_COUNTER
    CELL_COUNTER += 1
    return f"cell-{CELL_COUNTER:03d}"


def md_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": next_cell_id(),
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "id": next_cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def build_cells() -> list[dict[str, Any]]:
    return [
        md_cell(
            """
            ## Objective

            This notebook looks at the current C++ two-tissue compartment solver from three angles:

            1. **Noiseless synthetic validation**: does the implementation recover the known truth when the model is correct?
            2. **Noise sensitivity**: how quickly do bias and worst-case parameter errors grow once TAC noise is injected?
            3. **Real scan diagnostics**: on the current reversible fit of `AA1`, are large residuals random, or do they cluster around suspicious parameter regimes and active bounds?

            The target question is not only whether the solver *converges*, but whether the resulting parameter maps look trustworthy enough to build on.
            """
        ),
        code_cell(
            """
            from __future__ import annotations

            import subprocess
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import seaborn as sns
            from IPython.display import Markdown, display

            sns.set_theme(context="talk", style="whitegrid")
            plt.rcParams["axes.spines.top"] = False
            plt.rcParams["axes.spines.right"] = False
            plt.rcParams["figure.figsize"] = (10, 6)
            """
        ),
        code_cell(
            """
            def find_repo_root(start: Path) -> Path:
                start = start.resolve()
                for candidate in (start, *start.parents):
                    if (candidate / ".git").exists():
                        return candidate
                raise RuntimeError(f"Could not locate repo root from {start}")


            ROOT = find_repo_root(Path.cwd())
            DATA = ROOT / "data"
            REAL_INPUT = DATA / "real_uit"
            REAL_FIT_REV = DATA / "real_uit_rev"
            SYN_REV = DATA / "synthetic_rev"
            SYN_REV_OUT = DATA / "synthetic_rev_out"
            SYN_IRR = DATA / "synthetic_irr"
            SYN_IRR_OUT = DATA / "synthetic_irr_out"
            TMP = ROOT / "tmp" / "jupyter-notebook" / "noise_sweep"
            TMP.mkdir(parents=True, exist_ok=True)

            SOLVER = ROOT / "build" / "tcm_fit"
            if not SOLVER.exists():
                raise FileNotFoundError(f"Missing solver executable: {SOLVER}")

            ROOT
            """
        ),
        code_cell(
            """
            def parameter_names(model: str) -> list[str]:
                return ["K1", "k2", "k3", "k4", "vB", "Ki"] if model == "rev" else ["K1", "k2", "k3", "vB", "Ki"]


            def load_fit(run_dir: Path, model: str) -> dict[str, np.ndarray]:
                fit = {
                    "rmse": np.load(run_dir / "rmse.npy"),
                    "mask": np.load(run_dir / "mask.npy").astype(bool),
                    "status": np.load(run_dir / "status.npy"),
                    "iterations": np.load(run_dir / "iterations.npy"),
                }
                for name in parameter_names(model):
                    fit[name] = np.load(run_dir / f"{name}.npy")
                return fit


            def format_value(value: object) -> str:
                if isinstance(value, (float, np.floating)):
                    if np.isnan(value):
                        return "nan"
                    return f"{float(value):.6g}"
                if isinstance(value, (int, np.integer)):
                    return str(int(value))
                return str(value)


            def format_table(rows: list[dict[str, object]], columns: list[str] | None = None) -> str:
                if not rows:
                    return ""
                columns = columns or list(rows[0].keys())
                widths = []
                for col in columns:
                    width = max(len(col), *(len(format_value(row.get(col, ""))) for row in rows))
                    widths.append(width)

                header = " | ".join(col.ljust(width) for col, width in zip(columns, widths))
                sep = "-+-".join("-" * width for width in widths)
                body = []
                for row in rows:
                    body.append(
                        " | ".join(format_value(row.get(col, "")).ljust(width) for col, width in zip(columns, widths))
                    )
                return "\\n".join([header, sep, *body])


            def show_table(rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
                display(Markdown("```text\\n" + format_table(rows, columns) + "\\n```"))


            def flatten_masked(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
                return np.asarray(arr[mask], dtype=np.float64)


            def quantiles(values: np.ndarray, probs: list[float]) -> dict[float, float]:
                return {prob: float(np.nanquantile(values, prob)) for prob in probs}


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


            def predict_irrev(time: np.ndarray, aif: np.ndarray, K1: float, k2: float, k3: float, vB: float) -> np.ndarray:
                lam = max(k2 + k3, 1e-12)
                z = 0.0
                icp = 0.0
                out = np.zeros_like(time, dtype=np.float64)
                if time[0] > 0:
                    z = exp_conv_step(0.0, lam, float(time[0]), 0.0, float(aif[0]))
                    icp = 0.5 * float(aif[0]) * float(time[0])
                out[0] = (1 - vB) * (K1 / lam * (k3 * icp + k2 * z)) + vB * aif[0]
                for i in range(len(time) - 1):
                    dt = float(time[i + 1] - time[i])
                    z = exp_conv_step(z, lam, dt, float(aif[i]), float(aif[i + 1]))
                    icp += 0.5 * float(aif[i] + aif[i + 1]) * dt
                    ct = K1 / lam * (k3 * icp + k2 * z)
                    out[i + 1] = (1 - vB) * ct + vB * aif[i + 1]
                return out


            def predict_rev(
                time: np.ndarray,
                aif: np.ndarray,
                K1: float,
                k2: float,
                k3: float,
                k4: float,
                vB: float,
            ) -> np.ndarray:
                rate_sum = k2 + k3 + k4
                disc = max(rate_sum**2 - 4.0 * k2 * k4, 0.0)
                root = np.sqrt(disc)
                a1 = 0.5 * (rate_sum + root)
                a2 = 0.5 * (rate_sum - root)
                denom = abs(a1 - a2)
                if denom < 1e-12:
                    denom = 1e-12
                c1 = K1 * (a1 - k3 - k4) / denom
                c2 = -K1 * (a2 - k3 - k4) / denom

                z1 = 0.0
                z2 = 0.0
                out = np.zeros_like(time, dtype=np.float64)
                if time[0] > 0:
                    z1 = exp_conv_step(0.0, a1, float(time[0]), 0.0, float(aif[0]))
                    z2 = exp_conv_step(0.0, a2, float(time[0]), 0.0, float(aif[0]))
                out[0] = (1 - vB) * (c1 * z1 + c2 * z2) + vB * aif[0]
                for i in range(len(time) - 1):
                    dt = float(time[i + 1] - time[i])
                    z1 = exp_conv_step(z1, a1, dt, float(aif[i]), float(aif[i + 1]))
                    z2 = exp_conv_step(z2, a2, dt, float(aif[i]), float(aif[i + 1]))
                    out[i + 1] = (1 - vB) * (c1 * z1 + c2 * z2) + vB * aif[i + 1]
                return out


            def parameter_error_rows(input_dir: Path, run_dir: Path, model: str) -> list[dict[str, object]]:
                fit = load_fit(run_dir, model)
                mask = fit["mask"]
                rows = []
                for name in parameter_names(model):
                    pred = fit[name]
                    truth = np.load(input_dir / f"truth_{name}.npy")
                    err = pred[mask] - truth[mask]
                    rows.append(
                        {
                            "parameter": name,
                            "mae": float(np.mean(np.abs(err))),
                            "bias": float(np.mean(err)),
                            "median_abs": float(np.median(np.abs(err))),
                            "p95_abs": float(np.quantile(np.abs(err), 0.95)),
                            "max_abs": float(np.max(np.abs(err))),
                        }
                    )
                return rows


            def collect_error_vectors(input_dir: Path, run_dir: Path, model: str) -> dict[str, np.ndarray]:
                fit = load_fit(run_dir, model)
                mask = fit["mask"]
                vectors = {}
                for name in parameter_names(model):
                    pred = fit[name]
                    truth = np.load(input_dir / f"truth_{name}.npy")
                    vectors[name] = pred[mask] - truth[mask]
                return vectors


            def ensure_noise_case(model: str, noise_std: float) -> tuple[Path, Path]:
                tag = f"{model}_noise_{noise_std:.3f}".replace(".", "p")
                input_dir = TMP / tag
                output_dir = TMP / f"{tag}_out"

                if not input_dir.exists():
                    subprocess.run(
                        [
                            "python3",
                            "scripts/make_synthetic_dataset.py",
                            str(input_dir),
                            "--model",
                            model,
                            "--noise-std",
                            str(noise_std),
                        ],
                        cwd=ROOT,
                        check=True,
                    )

                if not output_dir.exists():
                    subprocess.run(
                        [
                            str(SOLVER),
                            "--input-dir",
                            str(input_dir),
                            "--output-dir",
                            str(output_dir),
                            "--model",
                            model,
                            "--mask-threshold",
                            "0.0",
                        ],
                        cwd=ROOT,
                        check=True,
                    )

                return input_dir, output_dir


            def pick_quantile_indices(values: np.ndarray, quantiles_to_pick: list[float]) -> list[int]:
                values = np.asarray(values)
                valid = np.isfinite(values)
                idx_valid = np.flatnonzero(valid)
                chosen = []
                for quantile in quantiles_to_pick:
                    target = float(np.quantile(values[valid], quantile))
                    local = int(np.argmin(np.abs(values[valid] - target)))
                    chosen.append(int(idx_valid[local]))
                return chosen


            def subsample(values: np.ndarray, max_points: int = 4000) -> np.ndarray:
                values = np.asarray(values)
                if values.size <= max_points:
                    return values
                idx = np.linspace(0, values.size - 1, max_points).astype(int)
                return values[idx]
            """
        ),
        md_cell(
            """
            ## 1. Noiseless Synthetic Validation

            The first question is whether the implementation itself is correct in the easy case. If it fails here, there is no point analyzing real data yet.
            """
        ),
        code_cell(
            """
            noiseless_rows = []
            noiseless_rows.extend(
                [{"model": "rev", **row} for row in parameter_error_rows(SYN_REV, SYN_REV_OUT, "rev")]
            )
            noiseless_rows.extend(
                [{"model": "irr", **row} for row in parameter_error_rows(SYN_IRR, SYN_IRR_OUT, "irr")]
            )
            show_table(
                noiseless_rows,
                columns=["model", "parameter", "mae", "bias", "median_abs", "p95_abs", "max_abs"],
            )

            rev_err = collect_error_vectors(SYN_REV, SYN_REV_OUT, "rev")
            irr_err = collect_error_vectors(SYN_IRR, SYN_IRR_OUT, "irr")

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            for ax, title, errors, transform in [
                (axes[0, 0], "Reversible Signed Error", rev_err, lambda x: x),
                (axes[0, 1], "Reversible Absolute Error", rev_err, np.abs),
                (axes[1, 0], "Irreversible Signed Error", irr_err, lambda x: x),
                (axes[1, 1], "Irreversible Absolute Error", irr_err, np.abs),
            ]:
                labels = []
                values = []
                for name, vec in errors.items():
                    sample = transform(vec)
                    labels.extend([name] * len(sample))
                    values.extend(sample.tolist())
                sns.boxplot(x=labels, y=values, ax=ax, showfliers=False)
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=30)
                ax.set_ylabel("error")
            plt.tight_layout()
            """
        ),
        md_cell(
            """
            ## 2. Noise Sensitivity And Bias

            Your core concern was bias under noisy TACs. To make that visible, the notebook runs a small Gaussian-noise sweep on the synthetic datasets and tracks mean bias, mean absolute error, and worst-case absolute error.
            """
        ),
        code_cell(
            """
            noise_levels = [0.0, 0.02, 0.05, 0.10]
            noise_rows = []
            for model in ["rev", "irr"]:
                for noise_std in noise_levels:
                    input_dir, output_dir = ensure_noise_case(model, noise_std)
                    for row in parameter_error_rows(input_dir, output_dir, model):
                        noise_rows.append({"model": model, "noise_std": noise_std, **row})

            show_table(
                [row for row in noise_rows if row["noise_std"] == max(noise_levels)],
                columns=["model", "noise_std", "parameter", "mae", "bias", "p95_abs", "max_abs"],
            )

            fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
            for row_idx, model in enumerate(["rev", "irr"]):
                model_rows = [row for row in noise_rows if row["model"] == model]
                params = [name for name in parameter_names(model) if name != "Ki"]
                for col_idx, metric in enumerate(["mae", "bias"]):
                    ax = axes[row_idx, col_idx]
                    for name in params:
                        ys = [row[metric] for row in model_rows if row["parameter"] == name]
                        xs = [row["noise_std"] for row in model_rows if row["parameter"] == name]
                        ax.plot(xs, ys, marker="o", label=name)
                    ax.set_title(f"{model.upper()} {metric.upper()} vs noise")
                    ax.set_xlabel("noise std")
                    ax.set_ylabel(metric)
                    ax.legend(ncol=3, fontsize=10)
            plt.tight_layout()

            fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
            for ax, model in zip(axes, ["rev", "irr"]):
                model_rows = [row for row in noise_rows if row["model"] == model]
                params = parameter_names(model)
                for name in params:
                    ys = [row["max_abs"] for row in model_rows if row["parameter"] == name]
                    xs = [row["noise_std"] for row in model_rows if row["parameter"] == name]
                    ax.plot(xs, ys, marker="o", label=name)
                ax.set_title(f"{model.upper()} worst-case absolute error")
                ax.set_xlabel("noise std")
                ax.set_ylabel("max_abs")
                ax.legend(ncol=3, fontsize=10)
            plt.tight_layout()
            """
        ),
        md_cell(
            """
            ## 3. Real Reversible Fit Diagnostics

            Here the question changes. There is no ground truth, so the useful diagnostics are:

            - residual RMSE and normalized RMSE,
            - convergence status,
            - concentration of voxels at parameter bounds,
            - and whether the worst residuals are enriched for those boundary hits.
            """
        ),
        code_cell(
            """
            real_image = np.load(REAL_INPUT / "image.npy")
            real_time = np.load(REAL_INPUT / "time.npy")
            real_aif = np.load(REAL_INPUT / "aif.npy")
            real = load_fit(REAL_FIT_REV, "rev")

            real_mask = real["mask"]
            real_vox = real_image.reshape(real_image.shape[0], -1)
            real_peak = real_vox.max(axis=0).reshape(real_mask.shape)
            real_auc = np.trapezoid(real_image, x=real_time, axis=0)
            real_rel_peak = np.where(real_mask, real["rmse"] / np.maximum(real_peak, 1e-6), np.nan)
            real_rel_auc = np.where(real_mask, real["rmse"] / np.maximum(real_auc, 1e-6), np.nan)

            rmse_vals = real["rmse"][real_mask]
            rel_peak_vals = real_rel_peak[real_mask]
            rel_auc_vals = real_rel_auc[real_mask]

            status_counts = {
                int(k): int(v)
                for k, v in zip(*np.unique(real["status"][real_mask], return_counts=True))
            }
            summary_rows = [
                {"metric": "masked_voxels", "value": int(real_mask.sum())},
                {"metric": "median_rmse", "value": float(np.median(rmse_vals))},
                {"metric": "p95_rmse", "value": float(np.quantile(rmse_vals, 0.95))},
                {"metric": "p99_rmse", "value": float(np.quantile(rmse_vals, 0.99))},
                {"metric": "max_rmse", "value": float(np.max(rmse_vals))},
                {"metric": "median_rel_peak_rmse", "value": float(np.median(rel_peak_vals))},
                {"metric": "p95_rel_peak_rmse", "value": float(np.quantile(rel_peak_vals, 0.95))},
                {"metric": "p99_rel_peak_rmse", "value": float(np.quantile(rel_peak_vals, 0.99))},
                {"metric": "max_rel_peak_rmse", "value": float(np.max(rel_peak_vals))},
                {"metric": "median_rel_auc_rmse", "value": float(np.median(rel_auc_vals))},
                {"metric": "p95_rel_auc_rmse", "value": float(np.quantile(rel_auc_vals, 0.95))},
                {"metric": "p99_rel_auc_rmse", "value": float(np.quantile(rel_auc_vals, 0.99))},
            ]
            show_table(summary_rows, columns=["metric", "value"])

            q95 = float(np.quantile(rmse_vals, 0.95))
            q99 = float(np.quantile(rmse_vals, 0.99))
            worst95 = real_mask & (real["rmse"] >= q95)
            worst99 = real_mask & (real["rmse"] >= q99)

            bounds = {"K1": 2.0, "k2": 3.0, "k3": 3.0, "k4": 3.0, "vB": 0.2}
            boundary_rows = []
            for name, ub in bounds.items():
                low = np.isclose(real[name], 0.0, atol=1e-8)
                high = np.isclose(real[name], ub, atol=1e-8)
                for label, active_mask in [("all", real_mask), ("worst_5pct", worst95), ("worst_1pct", worst99)]:
                    boundary_rows.append(
                        {
                            "parameter": name,
                            "bucket": label,
                            "low_frac": float(np.mean(low[active_mask])),
                            "high_frac": float(np.mean(high[active_mask])),
                        }
                    )
            show_table(boundary_rows, columns=["parameter", "bucket", "low_frac", "high_frac"])

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            sns.histplot(rmse_vals, bins=80, ax=axes[0], color="#1f77b4")
            axes[0].axvline(q95, color="black", linestyle="--", label="95th")
            axes[0].axvline(q99, color="red", linestyle="--", label="99th")
            axes[0].set_title("RMSE distribution")
            axes[0].legend()

            sns.histplot(rel_peak_vals, bins=80, ax=axes[1], color="#ff7f0e")
            axes[1].set_title("Relative peak RMSE distribution")
            axes[1].set_xlabel("RMSE / peak TAC")

            sns.barplot(x=list(status_counts.keys()), y=list(status_counts.values()), ax=axes[2], color="#2ca02c")
            axes[2].set_title("Convergence status counts")
            axes[2].set_xlabel("status code")
            axes[2].set_ylabel("voxels")
            plt.tight_layout()
            """
        ),
        code_cell(
            """
            def quartile_boxplot(ax: plt.Axes, parameter: np.ndarray, metric: np.ndarray, label: str) -> None:
                values = parameter[real_mask]
                metric_values = metric[real_mask]
                edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
                if np.unique(edges).size < 4:
                    ax.text(0.5, 0.5, "Too few unique values", ha="center", va="center")
                    ax.set_title(label)
                    return

                bins = np.digitize(values, edges[1:-1], right=True)
                labels = []
                plot_values = []
                for idx in range(4):
                    sample = subsample(metric_values[bins == idx], max_points=4000)
                    labels.extend([f"Q{idx+1}"] * len(sample))
                    plot_values.extend(sample.tolist())
                sns.boxplot(x=labels, y=plot_values, ax=ax, showfliers=False)
                ax.set_title(label)
                ax.set_xlabel("parameter quartile")
                ax.set_ylabel("RMSE / peak TAC")


            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            for ax, name in zip(axes.flat, ["K1", "k2", "k3", "k4", "vB", "Ki"]):
                quartile_boxplot(ax, real[name], real_rel_peak, name)
            plt.tight_layout()
            """
        ),
        code_cell(
            """
            flat_rmse = real["rmse"].reshape(-1)
            flat_rel_peak = real_rel_peak.reshape(-1)
            voxel_ids = pick_quantile_indices(flat_rel_peak[real_mask.reshape(-1)], [0.05, 0.25, 0.5, 0.9, 0.99, 1.0])
            valid_ids = np.flatnonzero(real_mask.reshape(-1))
            chosen_linear = [int(valid_ids[idx]) for idx in voxel_ids]

            fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=False)
            for ax, linear_idx, q in zip(axes.flat, chosen_linear, [0.05, 0.25, 0.5, 0.9, 0.99, 1.0]):
                z, y, x = np.unravel_index(linear_idx, real_mask.shape)
                tac = real_vox[:, linear_idx]
                pred = predict_rev(
                    real_time,
                    real_aif,
                    float(real["K1"][z, y, x]),
                    float(real["k2"][z, y, x]),
                    float(real["k3"][z, y, x]),
                    float(real["k4"][z, y, x]),
                    float(real["vB"][z, y, x]),
                )
                ax.plot(real_time, tac, marker="o", linewidth=2, label="measured")
                ax.plot(real_time, pred, linewidth=2, label="fit")
                ax.set_title(
                    f"q={q:.2f}, voxel=({z},{y},{x})\\n"
                    f"rmse={real['rmse'][z,y,x]:.3f}, rel={real_rel_peak[z,y,x]:.3f}, status={int(real['status'][z,y,x])}"
                )
                ax.set_xlabel("time [min]")
                ax.set_ylabel("activity")
            axes[0, 0].legend()
            plt.tight_layout()
            """
        ),
        code_cell(
            """
            slice_scores = np.nanmean(real_rel_peak, axis=(1, 2))
            slice_idx = int(np.nanargmax(slice_scores))

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            im0 = axes[0].imshow(real["rmse"][slice_idx], cmap="magma")
            axes[0].set_title(f"RMSE slice z={slice_idx}")
            plt.colorbar(im0, ax=axes[0], fraction=0.046)

            im1 = axes[1].imshow(real_rel_peak[slice_idx], cmap="viridis", vmin=0, vmax=np.nanquantile(real_rel_peak[real_mask], 0.99))
            axes[1].set_title("Relative peak RMSE")
            plt.colorbar(im1, ax=axes[1], fraction=0.046)

            im2 = axes[2].imshow(np.isclose(real["vB"][slice_idx], 0.2, atol=1e-8), cmap="gray_r")
            axes[2].set_title("vB at upper bound")
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            im3 = axes[3].imshow(np.isclose(real["k4"][slice_idx], 0.0, atol=1e-8), cmap="gray_r")
            axes[3].set_title("k4 at lower bound")
            plt.colorbar(im3, ax=axes[3], fraction=0.046)
            plt.tight_layout()
            """
        ),
        code_cell(
            '''
            max_noise = max(noise_levels)
            worst_noise_rows = [row for row in noise_rows if row["noise_std"] == max_noise]
            rev_noise = {row["parameter"]: row for row in worst_noise_rows if row["model"] == "rev"}
            irr_noise = {row["parameter"]: row for row in worst_noise_rows if row["model"] == "irr"}

            assessment = f"""
            ### Assessment

            **Implementation correctness**

            - On noiseless synthetic data the solver is numerically consistent with the model. Parameter errors are very small, so the C++ forward model and optimizer do not show obvious implementation bugs.

            **Noise sensitivity**

            - In the reversible noise sweep at noise std = {max_noise:.2f}, the largest mean absolute errors were:
              - `k2`: {rev_noise['k2']['mae']:.4g}
              - `k3`: {rev_noise['k3']['mae']:.4g}
              - `k4`: {rev_noise['k4']['mae']:.4g}
            - In the irreversible noise sweep at noise std = {max_noise:.2f}, the largest mean absolute errors were:
              - `k2`: {irr_noise['k2']['mae']:.4g}
              - `k3`: {irr_noise['k3']['mae']:.4g}
            - Bias stays much smaller than worst-case error in this small sweep, which points more toward variance / identifiability problems than a gross deterministic implementation bias.

            **Real reversible fit**

            - Median RMSE is {np.median(rmse_vals):.3f}; the 95th and 99th percentiles are {np.quantile(rmse_vals, 0.95):.3f} and {np.quantile(rmse_vals, 0.99):.3f}.
            - Median normalized RMSE is {np.median(rel_peak_vals):.3f} of the voxel peak TAC, and the 99th percentile is {np.quantile(rel_peak_vals, 0.99):.3f}.
            - Structural warning signs are visible in the fitted parameters:
              - `vB` is at the lower bound in {np.mean(np.isclose(real['vB'][real_mask], 0.0, atol=1e-8)) * 100:.1f}% of fitted voxels.
              - `k4` is at the lower bound in {np.mean(np.isclose(real['k4'][real_mask], 0.0, atol=1e-8)) * 100:.1f}% of fitted voxels.
              - In the worst 1% RMSE voxels, `vB` hits its upper bound in {np.mean(np.isclose(real['vB'][worst99], 0.2, atol=1e-8)) * 100:.1f}% of cases.
              - In the worst 1% RMSE voxels, `K1` hits its upper bound in {np.mean(np.isclose(real['K1'][worst99], 2.0, atol=1e-8)) * 100:.1f}% of cases.

            **Bottom line**

            - The solver itself looks good enough to continue with.
            - The real-data fit does **not** look fully healthy yet. The concentration of bad fits at active bounds suggests structural problems such as parameter non-identifiability, model mismatch, insufficient weighting, or overly wide / poorly calibrated bounds.
            - Before using these maps as training targets, I would tighten the model-selection and validation loop: add frame-average modeling if needed, calibrate weights, and consider fallback to a simpler model or constrained restart logic in voxels that saturate multiple bounds.
            """

            display(Markdown(assessment))
            '''
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate the PET fit analysis notebook.")
    parser.add_argument("notebook", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.notebook.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    notebook["cells"] = build_cells()

    with args.notebook.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
