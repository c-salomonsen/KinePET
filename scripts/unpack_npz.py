from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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

    arr = np.load(args.npz, allow_pickle=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(arr[args.image_key], dtype=np.float64)
    time = np.asarray(arr[args.time_key], dtype=np.float64)
    aif = np.asarray(arr[args.aif_key], dtype=np.float64)

    if image.ndim != 4:
        raise ValueError("image must have shape (time, depth, height, width)")
    if time.ndim != 1 or aif.ndim != 1:
        raise ValueError("time and aif must be 1D vectors")
    if image.shape[0] != time.shape[0] or time.shape[0] != aif.shape[0]:
        raise ValueError("time dimension mismatch between image, time, and aif")

    np.save(args.output_dir / "image.npy", image)
    np.save(args.output_dir / "time.npy", time)
    np.save(args.output_dir / "aif.npy", aif)

    metadata = {}
    for key, out_name in (
        (args.sample_id_key, "sample_id"),
        (args.group_key, "group"),
    ):
        if key in arr:
            value = arr[key]
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            metadata[out_name] = value

    if metadata:
        with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
