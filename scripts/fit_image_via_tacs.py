from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-oriented Python pipeline that preprocesses and flattens PET "
            "data to TAC batches, calls tcm_fit_tacs, then rebuilds parameter maps."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tac-fit", type=Path, default=Path("build/tcm_fit_tacs"))
    parser.add_argument("--model", choices=["rev", "irr"], required=True)
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
    parser.add_argument("--weights-file", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--no-linear-warm-start", action="store_true")
    return parser.parse_args()


def ensure_time_minutes(time_minutes: np.ndarray) -> np.ndarray:
    out = np.asarray(time_minutes, dtype=np.float64).reshape(-1)
    if np.max(out) > 60.0:
        out = out / 60.0
    return out


def build_mask(
    image: np.ndarray,
    threshold: float,
    keep_largest_component: bool,
    fill_holes: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    peak = image.max(axis=0)
    raw_mask = peak >= threshold
    cleaned_mask = raw_mask.copy()
    removed_mask = np.zeros_like(raw_mask, dtype=bool)
    filled_mask = np.zeros_like(raw_mask, dtype=bool)

    if keep_largest_component and np.any(cleaned_mask):
        structure = ndimage.generate_binary_structure(3, 1)
        labels, n_labels = ndimage.label(cleaned_mask, structure=structure)
        if n_labels > 0:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            keep_label = int(np.argmax(counts))
            largest = labels == keep_label
            removed_mask = cleaned_mask & ~largest
            cleaned_mask = largest

    if fill_holes and np.any(cleaned_mask):
        filled = ndimage.binary_fill_holes(cleaned_mask)
        filled_mask = filled & ~cleaned_mask
        cleaned_mask = filled

    return (
        raw_mask.astype(np.int32),
        cleaned_mask.astype(np.int32),
        removed_mask.astype(np.int32),
        filled_mask.astype(np.int32),
    )


def apply_prefit_preprocessing(
    image: np.ndarray,
    spatial_enable: bool,
    spatial_radius: int,
    spatial_sigma: float,
    temporal_filter: str,
    temporal_radius: int,
    temporal_gaussian_sigma: float,
) -> np.ndarray:
    fit_image = np.asarray(image, dtype=np.float64).copy()

    if spatial_enable:
        truncate = max(float(spatial_radius) / max(spatial_sigma, 1e-12), 0.5)
        fit_image = ndimage.gaussian_filter(
            fit_image,
            sigma=(0.0, spatial_sigma, spatial_sigma, spatial_sigma),
            mode="nearest",
            truncate=truncate,
        )

    if temporal_filter == "median":
        fit_image = ndimage.median_filter(
            fit_image,
            size=(2 * temporal_radius + 1, 1, 1, 1),
            mode="nearest",
        )
    elif temporal_filter == "gaussian":
        truncate = max(float(temporal_radius) / max(temporal_gaussian_sigma, 1e-12), 0.5)
        fit_image = ndimage.gaussian_filter(
            fit_image,
            sigma=(temporal_gaussian_sigma, 0.0, 0.0, 0.0),
            mode="nearest",
            truncate=truncate,
        )

    return fit_image


def compute_ki(K1: float, k2: float, k3: float) -> float:
    denom = k2 + k3
    if denom <= 1e-12:
        return 0.0
    return K1 * k3 / denom


def selective_status_median_postfilter(
    model: str,
    mask: np.ndarray,
    status: np.ndarray,
    maps: dict[str, np.ndarray],
    radius: int,
    min_reliable_neighbors: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    target_mask = ((status == 2) | (status == 4)) & mask
    donor_mask = ((status == 1) | (status == 3)) & mask
    applied_mask = np.zeros_like(mask, dtype=bool)
    unresolved_mask = np.zeros_like(mask, dtype=bool)

    updated = {name: value.copy() for name, value in maps.items()}
    for z, y, x in np.argwhere(target_mask):
        z0 = max(0, z - radius)
        y0 = max(0, y - radius)
        x0 = max(0, x - radius)
        z1 = min(mask.shape[0], z + radius + 1)
        y1 = min(mask.shape[1], y + radius + 1)
        x1 = min(mask.shape[2], x + radius + 1)

        donors = donor_mask[z0:z1, y0:y1, x0:x1]
        if int(donors.sum()) < min_reliable_neighbors:
            unresolved_mask[z, y, x] = True
            continue

        donor_values = {}
        for name, arr in maps.items():
            values = arr[z0:z1, y0:y1, x0:x1][donors]
            values = values[np.isfinite(values)]
            donor_values[name] = values

        if any(values.size < min_reliable_neighbors for values in donor_values.values()):
            unresolved_mask[z, y, x] = True
            continue

        for name, values in donor_values.items():
            updated[name][z, y, x] = float(np.median(values))

        updated["Ki"][z, y, x] = compute_ki(
            float(updated["K1"][z, y, x]),
            float(updated["k2"][z, y, x]),
            float(updated["k3"][z, y, x]),
        )
        applied_mask[z, y, x] = True

    return (
        updated,
        target_mask.astype(np.int32),
        applied_mask.astype(np.int32),
        unresolved_mask.astype(np.int32),
    )


def flatten_masked_aif(aif: np.ndarray, mask: np.ndarray) -> np.ndarray:
    aif = np.asarray(aif, dtype=np.float64)
    n_mask = int(mask.sum())
    if aif.ndim == 1:
        return aif
    if aif.ndim != 2:
        raise ValueError("aif.npy must be 1D or 2D for the image-via-tacs pipeline")
    if aif.shape[1] == 1:
        return aif
    if aif.shape[1] == mask.size:
        return aif[:, mask.reshape(-1)]
    if aif.shape[1] == n_mask:
        return aif
    raise ValueError("2D aif.npy must have second dimension 1, D*H*W, or the masked voxel count")


def flatten_masked_weights(weights_path: Path | None, mask: np.ndarray, t_count: int) -> np.ndarray | None:
    if weights_path is None:
        return None
    weights = np.load(weights_path)
    weights = np.asarray(weights, dtype=np.float64)
    n_mask = int(mask.sum())
    if weights.ndim == 1:
        if weights.shape[0] != t_count:
            raise ValueError("1D weights must have length T")
        return weights
    if weights.ndim != 2:
        raise ValueError("weights.npy must be 1D or 2D for the image-via-tacs pipeline")
    if weights.shape[0] != t_count:
        raise ValueError("weights.npy first dimension must equal T")
    if weights.shape[1] == 1:
        return weights
    if weights.shape[1] == mask.size:
        return weights[:, mask.reshape(-1)]
    if weights.shape[1] == n_mask:
        return weights
    raise ValueError("2D weights.npy must have second dimension 1, D*H*W, or the masked voxel count")


def save_run_metadata(path: Path, args: argparse.Namespace, mask_counts: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("mode=image_via_tacs\n")
        handle.write(f"model={'2tcm-reversible' if args.model == 'rev' else '2tcm-irreversible'}\n")
        handle.write(f"fit_vb={args.fit_vb}\n")
        handle.write(f"fixed_vb={args.fixed_vb}\n")
        handle.write(f"fit_delay={args.fit_delay}\n")
        handle.write(f"fixed_delay={args.fixed_delay}\n")
        handle.write(f"fit_dispersion={args.fit_dispersion}\n")
        handle.write(f"fixed_dispersion={args.fixed_dispersion}\n")
        handle.write(f"mask_threshold={args.mask_threshold}\n")
        handle.write(f"mask_keep_largest_component={args.mask_keep_largest_component}\n")
        handle.write(f"mask_fill_holes={args.mask_fill_holes}\n")
        handle.write(f"spatial_gaussian_prefilter={args.spatial_gaussian_prefilter}\n")
        handle.write(f"spatial_gaussian_radius={args.spatial_gaussian_radius}\n")
        handle.write(f"spatial_gaussian_sigma={args.spatial_gaussian_sigma}\n")
        handle.write(f"temporal_filter={args.temporal_filter}\n")
        handle.write(f"temporal_filter_radius={args.temporal_filter_radius}\n")
        handle.write(f"temporal_gaussian_sigma={args.temporal_gaussian_sigma}\n")
        handle.write(f"postfilter_status_median={args.postfilter_status_median}\n")
        handle.write(f"postfilter_radius={args.postfilter_radius}\n")
        handle.write(f"postfilter_min_reliable_neighbors={args.postfilter_min_reliable_neighbors}\n")
        handle.write("postfilter_target_statuses=2,4\n")
        handle.write("postfilter_donor_statuses=1,3\n")
        handle.write("rescue_refit=0\n")
        handle.write(f"max_iterations={args.max_iter}\n")
        handle.write(f"use_linear_warm_start={0 if args.no_linear_warm_start else 1}\n")
        handle.write(f"raw_mask_voxels={mask_counts['raw']}\n")
        handle.write(f"cleaned_mask_voxels={mask_counts['cleaned']}\n")
        handle.write(f"fitted_voxels={mask_counts['cleaned']}\n")
        handle.write(f"total_voxels={mask_counts['total']}\n")


def main() -> None:
    args = parse_args()
    if not args.tac_fit.exists():
        raise SystemExit(f"Missing TAC solver binary: {args.tac_fit}")

    image = np.load(args.input_dir / "image.npy")
    time = ensure_time_minutes(np.load(args.input_dir / "time.npy"))
    aif = np.load(args.input_dir / "aif.npy")
    if image.ndim != 4:
        raise SystemExit("image.npy must have shape (T, D, H, W)")
    if time.ndim != 1:
        raise SystemExit("time.npy must have shape (T,)")
    if image.shape[0] != time.shape[0]:
        raise SystemExit("time.npy length must match image.npy first dimension")
    if aif.shape[0] != time.shape[0]:
        raise SystemExit("aif.npy first dimension must match time.npy length")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_mask, mask, removed_mask, filled_mask = build_mask(
        image,
        threshold=args.mask_threshold,
        keep_largest_component=bool(args.mask_keep_largest_component),
        fill_holes=bool(args.mask_fill_holes),
    )
    mask_bool = mask.astype(bool)

    fit_image = apply_prefit_preprocessing(
        image,
        spatial_enable=bool(args.spatial_gaussian_prefilter),
        spatial_radius=args.spatial_gaussian_radius,
        spatial_sigma=args.spatial_gaussian_sigma,
        temporal_filter=args.temporal_filter,
        temporal_radius=args.temporal_filter_radius,
        temporal_gaussian_sigma=args.temporal_gaussian_sigma,
    )

    flat_tacs = fit_image[:, mask_bool]
    flat_aif = flatten_masked_aif(aif, mask_bool)
    flat_weights = flatten_masked_weights(args.weights_file, mask_bool, image.shape[0])

    with tempfile.TemporaryDirectory(prefix="tcm_tac_fit_") as tmp_name:
        tmp_dir = Path(tmp_name)
        input_dir = tmp_dir / "input"
        fit_dir = tmp_dir / "fit"
        input_dir.mkdir(parents=True, exist_ok=True)
        fit_dir.mkdir(parents=True, exist_ok=True)

        np.save(input_dir / "tacs.npy", np.ascontiguousarray(flat_tacs, dtype=np.float64))
        np.save(input_dir / "time.npy", time.astype(np.float64))
        np.save(input_dir / "aif.npy", np.ascontiguousarray(flat_aif, dtype=np.float64))
        if flat_weights is not None:
            np.save(input_dir / "weights.npy", np.ascontiguousarray(flat_weights, dtype=np.float64))

        command = [
            str(args.tac_fit),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(fit_dir),
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
            "--threads",
            str(args.threads),
            "--max-iter",
            str(args.max_iter),
        ]
        if flat_weights is not None:
            command.extend(["--weights-file", str(input_dir / "weights.npy")])
        if args.no_linear_warm_start:
            command.append("--no-linear-warm-start")
        subprocess.run(command, check=True)

        shape = image.shape[1:]
        indexer = np.flatnonzero(mask_bool.reshape(-1))

        def rebuild_scalar_map(values: np.ndarray, fill_value: float) -> np.ndarray:
            out = np.full(mask_bool.size, fill_value, dtype=np.float64)
            out[indexer] = values.reshape(-1)
            return out.reshape(shape)

        def rebuild_label_map(values: np.ndarray, fill_value: int) -> np.ndarray:
            out = np.full(mask_bool.size, fill_value, dtype=np.int32)
            out[indexer] = values.reshape(-1).astype(np.int32)
            return out.reshape(shape)

        raw_maps = {
            "K1": rebuild_scalar_map(np.load(fit_dir / "K1.npy"), np.nan),
            "k2": rebuild_scalar_map(np.load(fit_dir / "k2.npy"), np.nan),
            "k3": rebuild_scalar_map(np.load(fit_dir / "k3.npy"), np.nan),
            "vB": rebuild_scalar_map(np.load(fit_dir / "vB.npy"), np.nan),
            "Ki": rebuild_scalar_map(np.load(fit_dir / "Ki.npy"), np.nan),
        }
        if args.model == "rev":
            raw_maps["k4"] = rebuild_scalar_map(np.load(fit_dir / "k4.npy"), np.nan)
        if args.fit_delay or abs(args.fixed_delay) > 1e-12:
            raw_maps["delay"] = rebuild_scalar_map(np.load(fit_dir / "delay.npy"), np.nan)
        if args.fit_dispersion or abs(args.fixed_dispersion) > 1e-12:
            raw_maps["dispersion"] = rebuild_scalar_map(np.load(fit_dir / "dispersion.npy"), np.nan)

        rmse_map = rebuild_scalar_map(np.load(fit_dir / "rmse.npy"), np.nan)
        status_map = rebuild_label_map(np.load(fit_dir / "status.npy"), 0)
        iterations_map = rebuild_label_map(np.load(fit_dir / "iterations.npy"), 0)

    for name, arr in raw_maps.items():
        np.save(args.output_dir / f"{name}_rawfit.npy", arr)

    final_maps = raw_maps
    if args.postfilter_status_median:
        final_maps, post_target, post_applied, post_unresolved = selective_status_median_postfilter(
            model=args.model,
            mask=mask_bool,
            status=status_map,
            maps=raw_maps,
            radius=args.postfilter_radius,
            min_reliable_neighbors=args.postfilter_min_reliable_neighbors,
        )
    else:
        zeros = np.zeros_like(mask, dtype=np.int32)
        post_target = zeros
        post_applied = zeros
        post_unresolved = zeros

    for name, arr in final_maps.items():
        np.save(args.output_dir / f"{name}.npy", arr)
    np.save(args.output_dir / "rmse.npy", rmse_map)
    np.save(args.output_dir / "status.npy", status_map)
    np.save(args.output_dir / "iterations.npy", iterations_map)
    np.save(args.output_dir / "mask_raw.npy", raw_mask)
    np.save(args.output_dir / "mask.npy", mask)
    np.save(args.output_dir / "mask_removed.npy", removed_mask)
    np.save(args.output_dir / "mask_filled.npy", filled_mask)
    np.save(args.output_dir / "postfilter_target.npy", post_target)
    np.save(args.output_dir / "postfilter_applied.npy", post_applied)
    np.save(args.output_dir / "postfilter_unresolved.npy", post_unresolved)
    np.save(args.output_dir / "rescue_target.npy", np.zeros_like(mask, dtype=np.int32))
    np.save(args.output_dir / "rescue_applied.npy", np.zeros_like(mask, dtype=np.int32))
    np.save(args.output_dir / "rescue_unresolved.npy", np.zeros_like(mask, dtype=np.int32))

    with (args.output_dir / "pipeline_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mode": "image_via_tacs",
                "tac_fit_binary": str(args.tac_fit.resolve()),
                "input_dir": str(args.input_dir.resolve()),
                "output_dir": str(args.output_dir.resolve()),
            },
            handle,
            indent=2,
        )

    save_run_metadata(
        args.output_dir / "run.txt",
        args,
        {
            "raw": int(raw_mask.sum()),
            "cleaned": int(mask.sum()),
            "total": int(mask.size),
        },
    )


if __name__ == "__main__":
    main()
