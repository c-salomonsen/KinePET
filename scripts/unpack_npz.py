from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _extract_scalar(value: object) -> object:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def unpack_npz_bundle(
    npz_path: Path,
    output_dir: Path,
    image_key: str = "image",
    time_key: str = "time",
    aif_key: str = "aif",
    sample_id_key: str = "sample_id",
    group_key: str = "group",
) -> dict[str, object]:
    arr = np.load(npz_path, allow_pickle=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(arr[image_key], dtype=np.float64)
    time = np.asarray(arr[time_key], dtype=np.float64)
    aif = np.asarray(arr[aif_key], dtype=np.float64)

    if image.ndim != 4:
        raise ValueError("image must have shape (time, depth, height, width)")
    if time.ndim != 1 or aif.ndim != 1:
        raise ValueError("time and aif must be 1D vectors")
    if image.shape[0] != time.shape[0] or time.shape[0] != aif.shape[0]:
        raise ValueError("time dimension mismatch between image, time, and aif")

    np.save(output_dir / "image.npy", image)
    np.save(output_dir / "time.npy", time)
    np.save(output_dir / "aif.npy", aif)

    metadata: dict[str, object] = {}
    for key, out_name in (
        (sample_id_key, "sample_id"),
        (group_key, "group"),
    ):
        if key in arr:
            metadata[out_name] = _extract_scalar(arr[key])

    if metadata:
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PET NPZ bundle into separate NPY files for the C++ solver."
    )
    parser.add_argument("npz", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--time-key", default="time")
    parser.add_argument("--aif-key", default="aif")
    parser.add_argument("--sample-id-key", default="sample_id")
    parser.add_argument("--group-key", default="group")
    args = parser.parse_args()

    unpack_npz_bundle(
        npz_path=args.npz,
        output_dir=args.output_dir,
        image_key=args.image_key,
        time_key=args.time_key,
        aif_key=args.aif_key,
        sample_id_key=args.sample_id_key,
        group_key=args.group_key,
    )


if __name__ == "__main__":
    main()
