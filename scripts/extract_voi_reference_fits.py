from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import pickle
import urllib.request
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import lls
import numpy as np
import kinepet


DEFAULT_PYPET_URL = (
    "https://raw.githubusercontent.com/Kuttner/pyPET/"
    "8d552908f5f2accb2d026f6c3081dbf5bb961096/Tissue_compartment_modeling.py"
)


@dataclass
class SampleCurves:
    sample_id: str
    npz_path: Path
    voi_path: Path
    group: str | None
    npz_sample_id: str | None
    time_minutes: np.ndarray
    aif: np.ndarray
    brain: np.ndarray
    myocardium: np.ndarray


@dataclass
class SkippedSample:
    sample_id: str
    npz_path: str
    voi_path: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match raw NPZ files to expert VOI pickles, extract Brain/Myocardium "
            "curves, and fit them with the PyPET reference code, the TAC-based solver, "
            "and the LLS implementation."
        )
    )
    parser.add_argument(
        "--raw-npz-glob",
        default="data/raw/*.npz",
    )
    parser.add_argument(
        "--voi-glob",
        default="/Users/csa047/uit/phd/datasets/pet/VOI_SUV/VOI_*.pkl",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/validation/voi_reference"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pypet-module", type=Path, default=None)
    parser.add_argument("--pypet-url", default=DEFAULT_PYPET_URL)
    return parser.parse_args()


def discover_pickles(pattern: str) -> list[Path]:
    import glob

    return sorted(Path(path) for path in glob.glob(pattern))


def sanitize_sample_id(source_path: Path) -> str:
    name = source_path.stem
    if name.startswith("VOI_"):
        return name[len("VOI_") :]
    return name


def load_paired_sample(npz_path: Path, voi_path: Path) -> SampleCurves:
    with np.load(npz_path, allow_pickle=True) as arr:
        if "time" not in arr or "aif" not in arr:
            raise KeyError("Missing NPZ keys: time or aif")
        time_minutes = np.asarray(arr["time"], dtype=np.float64)
        aif = np.asarray(arr["aif"], dtype=np.float64)
        npz_sample_id = str(arr["sample_id"].item()) if "sample_id" in arr else None
        group = str(arr["group"].item()) if "group" in arr else None

    with voi_path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            loaded = pickle.load(handle)

    if "VOI_t" not in loaded or "VOI_A" not in loaded:
        raise KeyError("Missing VOI_t or VOI_A")

    voi_a = loaded["VOI_A"]
    missing = [name for name in ["Brain", "Myocardium"] if name not in voi_a]
    if missing:
        raise KeyError(f"Missing VOI_A keys: {', '.join(missing)}")

    voi_t = np.asarray(loaded["VOI_t"], dtype=np.float64)
    brain = np.asarray(voi_a["Brain"], dtype=np.float64)
    myocardium = np.asarray(voi_a["Myocardium"], dtype=np.float64)

    if time_minutes.ndim != 1 or aif.ndim != 1 or voi_t.ndim != 1 or brain.ndim != 1 or myocardium.ndim != 1:
        raise ValueError("Expected 1D time, aif, and VOI curves")
    if not (time_minutes.shape == aif.shape == brain.shape == myocardium.shape == voi_t.shape):
        raise ValueError("NPZ time/aif and VOI curves do not have matching lengths")

    if np.max(time_minutes) > 60.0:
        time_minutes = time_minutes / 60.0
    if np.max(voi_t) > 60.0:
        voi_t = voi_t / 60.0
    if not np.allclose(time_minutes, voi_t, atol=5e-4, rtol=0.0):
        raise ValueError("NPZ time vector and VOI_t do not match")

    return SampleCurves(
        sample_id=sanitize_sample_id(npz_path),
        npz_path=npz_path.resolve(),
        voi_path=voi_path.resolve(),
        group=group,
        npz_sample_id=npz_sample_id,
        time_minutes=time_minutes,
        aif=aif,
        brain=brain,
        myocardium=myocardium,
    )


def ensure_pypet_module(args: argparse.Namespace) -> Path:
    if args.pypet_module is not None:
        if not args.pypet_module.exists():
            raise SystemExit(f"Missing PyPET module: {args.pypet_module}")
        return args.pypet_module

    cache_dir = args.output_dir / "reference_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    module_path = cache_dir / "Tissue_compartment_modeling.py"
    if not module_path.exists():
        with urllib.request.urlopen(args.pypet_url) as response:
            module_path.write_bytes(response.read())
    return module_path


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("pypet_reference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_with_pypet(module, sample: SampleCurves, region: str, model: str) -> dict[str, object]:
    ct = sample.brain if region == "Brain" else sample.myocardium
    cp = sample.aif
    time = sample.time_minutes

    func = module.twoTCMrev if model == "rev" else module.twoTCMirrev
    try:
        with redirect_stdout(io.StringIO()):
            params, ki, _fit_curve, fit_error = func(
                cp.copy(),
                time.copy(),
                ct.copy(),
                INTERPOLATE=True,
                TIME_FRAMES=2.5,
                fit_vB=True,
            )
        params = np.asarray(params, dtype=np.float64)
        if model == "rev":
            K1, k2, vB, k3, k4 = params
        else:
            K1, k2, vB, k3 = params
            k4 = 0.0
        return {
            "sample_id": sample.sample_id,
            "npz_sample_id": sample.npz_sample_id or "",
            "group": sample.group or "",
            "region": region,
            "model": model,
            "method": "pypet",
            "K1": float(K1),
            "k2": float(k2),
            "k3": float(k3),
            "k4": float(k4),
            "vB": float(vB),
            "Ki": float(ki),
            "error_value": float(fit_error),
            "error_kind": "rss_interp_grid",
            "status": "success",
            "iterations": "",
            "npz_path": str(sample.npz_path),
            "voi_path": str(sample.voi_path),
        }
    except Exception as exc:
        return {
            "sample_id": sample.sample_id,
            "npz_sample_id": sample.npz_sample_id or "",
            "group": sample.group or "",
            "region": region,
            "model": model,
            "method": "pypet",
            "K1": np.nan,
            "k2": np.nan,
            "k3": np.nan,
            "k4": np.nan,
            "vB": np.nan,
            "Ki": np.nan,
            "error_value": np.nan,
            "error_kind": "rss_interp_grid",
            "status": f"error: {exc}",
            "iterations": "",
            "npz_path": str(sample.npz_path),
            "voi_path": str(sample.voi_path),
        }


def lls_error_rows(sample: SampleCurves, model: str, message: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region in ["Brain", "Myocardium"]:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "npz_sample_id": sample.npz_sample_id or "",
                "group": sample.group or "",
                "region": region,
                "model": model,
                "method": "lls",
                "K1": np.nan,
                "k2": np.nan,
                "k3": np.nan,
                "k4": np.nan,
                "vB": np.nan,
                "Ki": np.nan,
                "error_value": np.nan,
                "error_kind": "mse_interp_grid",
                "status": f"error: {message}",
                "iterations": "",
                "npz_path": str(sample.npz_path),
                "voi_path": str(sample.voi_path),
            }
        )
    return rows


def fit_with_lls(sample: SampleCurves, model: str) -> list[dict[str, object]]:
    tacs = np.stack([sample.brain, sample.myocardium], axis=1)
    model_name = "4k_vB" if model == "rev" else "3k_vB"

    try:
        kinetic_params, _fitted_tacs, mse = lls.fit_compartment_model(
            tacs=tacs,
            aif=sample.aif,
            time=sample.time_minutes,
            model=model_name,
            time_unit="min",
            interpolate=True,
            frame_duration=2.5 / 60.0,
        )
    except Exception as exc:
        return lls_error_rows(sample, model, str(exc))

    params = np.asarray(kinetic_params, dtype=np.float64)
    mse = np.asarray(mse, dtype=np.float64).reshape(-1)
    if params.ndim != 2 or params.shape[1] != 2:
        return lls_error_rows(
            sample,
            model,
            f"Unexpected kinetic parameter shape: {tuple(params.shape)}",
        )
    if mse.shape != (2,):
        return lls_error_rows(sample, model, f"Unexpected mse shape: {tuple(mse.shape)}")

    rows: list[dict[str, object]] = []
    for region, voxel in [("Brain", 0), ("Myocardium", 1)]:
        if model == "rev":
            K1, k2, k3, k4, vB, Ki = params[:, voxel]
        else:
            K1, k2, k3, vB, Ki = params[:, voxel]
            k4 = 0.0
        rows.append(
            {
                "sample_id": sample.sample_id,
                "npz_sample_id": sample.npz_sample_id or "",
                "group": sample.group or "",
                "region": region,
                "model": model,
                "method": "lls",
                "K1": float(K1),
                "k2": float(k2),
                "k3": float(k3),
                "k4": float(k4),
                "vB": float(vB),
                "Ki": float(Ki),
                "error_value": float(mse[voxel]),
                "error_kind": "mse_interp_grid",
                "status": "success",
                "iterations": "",
                "npz_path": str(sample.npz_path),
                "voi_path": str(sample.voi_path),
            }
        )
    return rows


def cpp_error_rows(sample: SampleCurves, model: str, message: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region in ["Brain", "Myocardium"]:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "npz_sample_id": sample.npz_sample_id or "",
                "group": sample.group or "",
                "region": region,
                "model": model,
                "method": "cpp",
                "K1": np.nan,
                "k2": np.nan,
                "k3": np.nan,
                "k4": np.nan,
                "vB": np.nan,
                "Ki": np.nan,
                "error_value": np.nan,
                "error_kind": "rmse_native_grid",
                "status": f"error: {message}",
                "iterations": "",
                "npz_path": str(sample.npz_path),
                "voi_path": str(sample.voi_path),
            }
        )
    return rows


def run_cpp_fit(args: argparse.Namespace, sample: SampleCurves, model: str) -> list[dict[str, object]]:
    tacs = np.stack([sample.brain, sample.myocardium], axis=1)
    try:
        fit = kinepet.fit_tacs(
            tacs,
            sample.time_minutes,
            sample.aif,
            model=model,
            fit_vb=True,
            threads=args.threads,
        )
    except Exception as exc:
        return cpp_error_rows(sample, model, str(exc))

    k4 = fit.k4 if fit.k4 is not None else np.zeros_like(fit.k3)
    rows: list[dict[str, object]] = []
    for region, voxel in [("Brain", 0), ("Myocardium", 1)]:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "npz_sample_id": sample.npz_sample_id or "",
                "group": sample.group or "",
                "region": region,
                "model": model,
                "method": "cpp",
                "K1": float(fit.K1[voxel]),
                "k2": float(fit.k2[voxel]),
                "k3": float(fit.k3[voxel]),
                "k4": float(k4[voxel]),
                "vB": float(fit.vB[voxel]),
                "Ki": float(fit.Ki[voxel]),
                "error_value": float(fit.rmse[voxel]),
                "error_kind": "rmse_native_grid",
                "status": str(int(fit.status[voxel])),
                "iterations": str(int(fit.iterations[voxel])),
                "npz_path": str(sample.npz_path),
                "voi_path": str(sample.voi_path),
            }
        )
    return rows


def save_curves_npz(output_dir: Path, samples: list[SampleCurves]) -> Path:
    output_path = output_dir / "voi_reference_curves.npz"
    np.savez(
        output_path,
        sample_id=np.array([sample.sample_id for sample in samples], dtype=object),
        npz_path=np.array([str(sample.npz_path) for sample in samples], dtype=object),
        voi_path=np.array([str(sample.voi_path) for sample in samples], dtype=object),
        group=np.array([sample.group or "" for sample in samples], dtype=object),
        npz_sample_id=np.array([sample.npz_sample_id or "" for sample in samples], dtype=object),
        time_minutes=np.stack([sample.time_minutes for sample in samples], axis=0),
        aif=np.stack([sample.aif for sample in samples], axis=0),
        brain=np.stack([sample.brain for sample in samples], axis=0),
        myocardium=np.stack([sample.myocardium for sample in samples], axis=0),
    )
    return output_path


def save_results_csv(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    output_path = output_dir / "voi_reference_fits.csv"
    fieldnames = [
        "sample_id",
        "npz_sample_id",
        "group",
        "region",
        "model",
        "method",
        "K1",
        "k2",
        "k3",
        "k4",
        "vB",
        "Ki",
        "error_value",
        "error_kind",
        "status",
        "iterations",
        "npz_path",
        "voi_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def save_skipped_csv(output_dir: Path, skipped: list[SkippedSample]) -> Path:
    output_path = output_dir / "voi_reference_skipped.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "npz_path", "voi_path", "reason"])
        writer.writeheader()
        for row in skipped:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "npz_path": row.npz_path,
                    "voi_path": row.voi_path,
                    "reason": row.reason,
                }
            )
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pypet_module_path = ensure_pypet_module(args)
    pypet_module = load_module(pypet_module_path)

    npz_paths = discover_pickles(args.raw_npz_glob)
    voi_paths = discover_pickles(args.voi_glob)
    if not npz_paths:
        raise SystemExit(f"No raw NPZ files matched: {args.raw_npz_glob}")
    if not voi_paths:
        raise SystemExit(f"No VOI pickle files matched: {args.voi_glob}")

    npz_by_id = {sanitize_sample_id(path): path for path in npz_paths}
    voi_by_id = {sanitize_sample_id(path): path for path in voi_paths}

    samples: list[SampleCurves] = []
    skipped: list[SkippedSample] = []

    for sample_id, npz_path in sorted(npz_by_id.items()):
        voi_path = voi_by_id.get(sample_id)
        if voi_path is None:
            skipped.append(
                SkippedSample(
                    sample_id=sample_id,
                    npz_path=str(npz_path.resolve()),
                    voi_path="",
                    reason="missing VOI pickle",
                )
            )
            continue
        try:
            samples.append(load_paired_sample(npz_path, voi_path))
        except Exception as exc:
            skipped.append(
                SkippedSample(
                    sample_id=sample_id,
                    npz_path=str(npz_path.resolve()),
                    voi_path=str(voi_path.resolve()),
                    reason=str(exc),
                )
            )

    for sample_id, voi_path in sorted(voi_by_id.items()):
        if sample_id in npz_by_id:
            continue
        skipped.append(
            SkippedSample(
                sample_id=sample_id,
                npz_path="",
                voi_path=str(voi_path.resolve()),
                reason="missing raw NPZ",
            )
        )

    rows: list[dict[str, object]] = []

    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.sample_id}")
        for model in ["irr", "rev"]:
            rows.extend(run_cpp_fit(args, sample, model))
            rows.extend(fit_with_lls(sample, model))
            for region in ["Brain", "Myocardium"]:
                rows.append(fit_with_pypet(pypet_module, sample, region, model))

    curves_path = save_curves_npz(args.output_dir, samples)
    csv_path = save_results_csv(args.output_dir, rows)
    skipped_path = save_skipped_csv(args.output_dir, skipped)
    metadata_path = args.output_dir / "metadata.json"
    metadata = {
        "raw_npz_glob": args.raw_npz_glob,
        "voi_glob": args.voi_glob,
        "raw_npz_count": len(npz_paths),
        "voi_pickle_count": len(voi_paths),
        "matched_sample_count": len(samples),
        "sample_count_fitted": len(samples),
        "sample_count_skipped": len(skipped),
        "curves_npz": str(curves_path.resolve()),
        "fits_csv": str(csv_path.resolve()),
        "skipped_csv": str(skipped_path.resolve()),
        "pypet_module": str(pypet_module_path.resolve()),
        "pypet_url": args.pypet_url if args.pypet_module is None else None,
        "cpp_solver": "kinepet Python binding",
        "cpp_threads": args.threads,
        "cpp_fit_vb": 1,
        "lls_package": str(Path(lls.__file__).resolve()),
        "lls_model_map": {"irr": "3k_vB", "rev": "4k_vB"},
        "lls_interpolate": True,
        "lls_frame_duration_seconds": 2.5,
        "pypet_interpolate": True,
        "pypet_time_frames_seconds": 2.5,
        "aif_source": "raw_npz.aif",
        "target_regions": ["Brain", "Myocardium"],
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote curves: {curves_path}")
    print(f"Wrote fits: {csv_path}")
    print(f"Wrote skipped: {skipped_path}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
