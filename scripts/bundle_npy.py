import csv
from dataclasses import dataclass, fields
from pathlib import Path
from shutil import copyfile
from typing import Any, Self, get_args

import numpy as np

model_to_parameters = {
    "2tcm-irreversible": ["K1", "k2", "k3"],
    "2tcm-reversible": ["K1", "k2", "k3", "k4"],
}


def resolve_type(value: str, target_type: type) -> object:
    if target_type is bool:
        return target_type(int(value))

    if target_type == tuple[int]:
        return tuple([int(x) for x in value.split(",")])

    if value.lower() == "none":
        return None

    # Resolve Union types by trying each option in order
    types = get_args(target_type)
    if len(types) > 0:
        for type in types:
            try:
                return resolve_type(value, type)
            except ValueError:
                continue

    return target_type(value)


@dataclass
class FittedMetadata:
    model: str
    fit_vb: bool
    fixed_vb: float
    fit_delay: bool
    fixed_delay: float
    fit_dispersion: bool
    fixed_dispersion: float
    mask_threshold: float
    mask_keep_largest_component: bool
    mask_fill_holes: bool
    spatial_gaussian_prefilter: bool
    spatial_gaussian_radius: int
    spatial_gaussian_sigma: float
    temporal_filter: str | None
    temporal_filter_radius: int | None
    temporal_gaussian_sigma: float | None
    raw_mask_voxels: int
    cleaned_mask_voxels: int
    postfilter_status_median: int
    postfilter_radius: int
    postfilter_min_reliable_neighbors: int
    postfilter_target_statuses: tuple[int]
    postfilter_donor_statuses: tuple[int]
    postfilter_targeted_voxels: int
    postfilter_applied_voxels: int
    postfilter_unresolved_voxels: int
    max_iterations: int
    use_linear_warm_start: bool
    fitted_voxels: int
    total_voxels: int

    @classmethod
    def load(cls, path: Path) -> Self:
        with open(path, "r") as f:
            metadata = {}
            field_name_type = {field.name: field.type for field in fields(cls)}

            for line in f:
                key, value = line.strip().split("=")

                assert key in {field.name for field in fields(cls)}, (
                    f"Unexpected metadata key: {key}"
                )

                metadata[key] = resolve_type(value, field_name_type[key])

        return cls(**metadata)

    def saveable_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class FittedSample:
    sample_id: str
    maps: dict[str, np.ndarray]
    metadata: FittedMetadata

    @classmethod
    def load(cls, path: Path, sample_id: str) -> Self:
        metadata = FittedMetadata.load(path / "run.txt")

        # Determine which parameters to load based on the model
        parameters = model_to_parameters[metadata.model]

        parameters.append("mask")  # Always include the mask

        # Add optional parameters if they were fitted
        if metadata.fit_vb:
            parameters.append("vB")
        if metadata.fit_delay:
            parameters.append("delay")
        if metadata.fit_dispersion:
            parameters.append("dispersion")

        maps = {}
        for param in parameters:
            map_path = path / f"{param}.npy"

            if not map_path.exists():
                raise FileNotFoundError(f"Expected parameter map not found: {map_path}")

            maps[param] = np.load(map_path)

        return cls(
            sample_id=str(sample_id),
            maps=maps,
            metadata=metadata,
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(path, sample_id=self.sample_id, metadata=self.metadata.saveable_dict(), **self.maps)


@dataclass
class FittedDataset:
    name: str
    path: Path
    samples: list[FittedSample]
    manifest: Path

    @classmethod
    def load(
        cls, path: Path, name: str | None = None, manifest_path: Path | None = None
    ) -> Self:
        samples = []

        manifest_file = manifest_path or (path / "batch_manifest.csv")
        assert manifest_file.exists(), f"Manifest file not found: {manifest_file}"
        assert manifest_file.suffix == ".csv", (
            f"Expected manifest file to be a CSV: {manifest_file}"
        )

        with open(manifest_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_dir_path = Path(row["fit_dir"])
                sample_id = row["sample_id"]
                sample = FittedSample.load(sample_dir_path, sample_id)
                samples.append(sample)

        return cls(
            name=name or path.name,
            path=path,
            samples=samples,
            manifest=manifest_file,
        )

    def save(self, path: Path) -> None:
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)

        # Save manifest file
        copyfile(self.manifest, path / "batch_manifest.csv")

        # Save each sample's parameter maps and metadata
        for sample in self.samples:
            sample.save(path / f"{sample.sample_id}.npz")

    def __repr__(self) -> str:
        return f"FittedDataset(name={self.name}, path={self.path}, samples={len(self.samples)})"


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Bundle fitted parameter maps and metadata into NPZ files"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the directory containing the fitted parameter maps and metadata",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to the directory where the bundled NPZ files will be saved",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional name for the dataset (defaults to the input directory name)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional path to the manifest CSV file (defaults to 'batch_manifest.csv' in the input directory)",
    )
    args = parser.parse_args()

    dataset = FittedDataset.load(
        path=args.input_path,
        name=args.name,
        manifest_path=args.manifest_path,
    )
    print(f"Loaded dataset: {dataset}")
    dataset.save(args.output_path)
    print(f"Saved bundled dataset to: {args.output_path}")
