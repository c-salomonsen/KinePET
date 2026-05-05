from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from unpack_npz import unpack_npz_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially unpack PET NPZ bundles and run the voxel-wise fitter into "
            "one per-sample output directory."
        )
    )
    parser.add_argument("--npz-glob", default="data/raw/*.npz")
    parser.add_argument("--unpacked-root", type=Path, default=Path("data/unpacked"))
    parser.add_argument("--fit-root", type=Path, default=Path("data/fits"))
    parser.add_argument("--tcm-fit", type=Path, default=Path("build/tcm_fit"))
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--time-key", default="time")
    parser.add_argument("--aif-key", default="aif")
    parser.add_argument("--sample-id-key", default="sample_id")
    parser.add_argument("--group-key", default="group")
    parser.add_argument("--model", choices=["rev", "irr"], default="rev")
    parser.add_argument("--fit-vb", type=int, choices=[0, 1], default=1)
    parser.add_argument("--fixed-vb", type=float, default=0.05)
    parser.add_argument("--fit-delay", type=int, choices=[0, 1], default=0)
    parser.add_argument("--fixed-delay", type=float, default=0.0)
    parser.add_argument("--fit-dispersion", type=int, choices=[0, 1], default=0)
    parser.add_argument("--fixed-dispersion", type=float, default=0.0)
    parser.add_argument("--mask-threshold", type=float, default=0.05)
    parser.add_argument("--mask-keep-largest-component", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mask-fill-holes", type=int, choices=[0, 1], default=1)
    parser.add_argument("--spatial-gaussian-prefilter", type=int, choices=[0, 1], default=0)
    parser.add_argument("--spatial-gaussian-radius", type=int, default=1)
    parser.add_argument("--spatial-gaussian-sigma", type=float, default=0.75)
    parser.add_argument("--temporal-filter", choices=["none", "median", "gaussian"], default="none")
    parser.add_argument("--temporal-filter-radius", type=int, default=1)
    parser.add_argument("--temporal-gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--postfilter-status-median", type=int, choices=[0, 1], default=1)
    parser.add_argument("--postfilter-radius", type=int, default=1)
    parser.add_argument("--postfilter-min-reliable-neighbors", type=int, default=3)
    parser.add_argument("--rescue-refit", type=int, choices=[0, 1], default=0)
    parser.add_argument("--rescue-radius", type=int, default=1)
    parser.add_argument("--rescue-min-reliable-neighbors", type=int, default=3)
    parser.add_argument("--rescue-iterations", type=int, default=12)
    parser.add_argument("--rescue-rmse-quantile", type=float, default=0.75)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--delete-unpacked-after-fit", action="store_true")
    parser.add_argument("--no-linear-warm-start", action="store_true")
    return parser.parse_args()


def extract_scalar(value: object) -> object:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def read_sample_id(npz_path: Path, sample_id_key: str) -> str | None:
    with np.load(npz_path, allow_pickle=True) as arr:
        if sample_id_key not in arr:
            return None
        value = extract_scalar(arr[sample_id_key])
        if value is None:
            return None
        text = str(value).strip()
        return text or None


def sanitize_sample_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "sample"


def discover_npz(npz_glob: str) -> list[Path]:
    return sorted(Path(path) for path in glob.glob(npz_glob))


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def build_solver_command(args: argparse.Namespace, input_dir: Path, output_dir: Path) -> list[str]:
    command = [
        str(args.tcm_fit),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--model",
        args.model,
        "--fit-vb",
        str(args.fit_vb),
        "--fixed-vb",
        str(args.fixed_vb),
        "--fit-delay",
        str(args.fit_delay),
        "--fixed-delay",
        str(args.fixed_delay),
        "--fit-dispersion",
        str(args.fit_dispersion),
        "--fixed-dispersion",
        str(args.fixed_dispersion),
        "--mask-threshold",
        str(args.mask_threshold),
        "--mask-keep-largest-component",
        str(args.mask_keep_largest_component),
        "--mask-fill-holes",
        str(args.mask_fill_holes),
        "--spatial-gaussian-prefilter",
        str(args.spatial_gaussian_prefilter),
        "--spatial-gaussian-radius",
        str(args.spatial_gaussian_radius),
        "--spatial-gaussian-sigma",
        str(args.spatial_gaussian_sigma),
        "--temporal-filter",
        args.temporal_filter,
        "--temporal-filter-radius",
        str(args.temporal_filter_radius),
        "--temporal-gaussian-sigma",
        str(args.temporal_gaussian_sigma),
        "--postfilter-status-median",
        str(args.postfilter_status_median),
        "--postfilter-radius",
        str(args.postfilter_radius),
        "--postfilter-min-reliable-neighbors",
        str(args.postfilter_min_reliable_neighbors),
        "--rescue-refit",
        str(args.rescue_refit),
        "--rescue-radius",
        str(args.rescue_radius),
        "--rescue-min-reliable-neighbors",
        str(args.rescue_min_reliable_neighbors),
        "--rescue-iterations",
        str(args.rescue_iterations),
        "--rescue-rmse-quantile",
        str(args.rescue_rmse_quantile),
        "--threads",
        str(args.threads),
        "--max-iter",
        str(args.max_iter),
    ]
    if args.no_linear_warm_start:
        command.append("--no-linear-warm-start")
    return command


def main() -> None:
    args = parse_args()
    npz_paths = discover_npz(args.npz_glob)
    if not npz_paths:
        raise SystemExit(f"No NPZ files matched: {args.npz_glob}")
    if not args.tcm_fit.exists():
        raise SystemExit(f"Missing solver binary: {args.tcm_fit}")

    args.unpacked_root.mkdir(parents=True, exist_ok=True)
    args.fit_root.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict[str, str]] = []
    seen_sample_names: dict[str, Path] = {}

    for npz_path in npz_paths:
        raw_sample_id = read_sample_id(npz_path, args.sample_id_key)
        sample_name = sanitize_sample_name(raw_sample_id or npz_path.stem)
        if sample_name in seen_sample_names and seen_sample_names[sample_name] != npz_path:
            raise SystemExit(
                f"Duplicate sample directory name '{sample_name}' for "
                f"{seen_sample_names[sample_name]} and {npz_path}"
            )
        seen_sample_names[sample_name] = npz_path

        unpack_dir = args.unpacked_root / sample_name
        fit_dir = args.fit_root / sample_name
        fit_done = (fit_dir / "run.txt").exists() and (fit_dir / "K1.npy").exists()

        print(f"[{sample_name}] source={npz_path}")

        metadata = unpack_npz_bundle(
            npz_path=npz_path,
            output_dir=unpack_dir,
            image_key=args.image_key,
            time_key=args.time_key,
            aif_key=args.aif_key,
            sample_id_key=args.sample_id_key,
            group_key=args.group_key,
        )

        write_json(
            unpack_dir / "source.json",
            {
                "npz_path": str(npz_path.resolve()),
                "sample_name": sample_name,
                "sample_id": metadata.get("sample_id", raw_sample_id or npz_path.stem),
                "group": metadata.get("group"),
            },
        )

        if args.skip_existing and fit_done:
            print(f"[{sample_name}] skipping fit because outputs already exist")
        else:
            fit_dir.mkdir(parents=True, exist_ok=True)
            solver_command = build_solver_command(args, unpack_dir, fit_dir)
            subprocess.run(solver_command, check=True)
            write_json(
                fit_dir / "input_metadata.json",
                {
                    "npz_path": str(npz_path.resolve()),
                    "unpacked_input_dir": str(unpack_dir.resolve()),
                    "sample_name": sample_name,
                    "sample_id": metadata.get("sample_id", raw_sample_id or npz_path.stem),
                    "group": metadata.get("group"),
                },
            )

        sample_rows.append(
            {
                "sample_name": sample_name,
                "sample_id": str(metadata.get("sample_id", raw_sample_id or npz_path.stem)),
                "group": str(metadata.get("group", "")),
                "npz_path": str(npz_path.resolve()),
                "unpacked_dir": str(unpack_dir.resolve()),
                "fit_dir": str(fit_dir.resolve()),
            }
        )

        if args.delete_unpacked_after_fit and (fit_dir / "run.txt").exists():
            shutil.rmtree(unpack_dir)
            print(f"[{sample_name}] deleted unpacked directory after successful fit")

    manifest_path = args.fit_root / "batch_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_name", "sample_id", "group", "npz_path", "unpacked_dir", "fit_dir"],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
