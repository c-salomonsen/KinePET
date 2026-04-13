from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize dynamic PET data, fitted parameter maps, and fit diagnostics in Napari."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing image.npy, time.npy, and aif.npy.",
    )
    parser.add_argument(
        "--fit-dir",
        type=Path,
        required=True,
        help="Directory containing fitted parameter .npy outputs.",
    )
    parser.add_argument(
        "--initial-voxel",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
        help="Initial voxel to inspect in z y x order.",
    )
    return parser.parse_args()


def read_run_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not path.exists():
        return metadata
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def robust_limits(array: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    values = np.asarray(array, dtype=np.float64)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= mask
    values = values[valid]
    if values.size == 0:
        return 0.0, 1.0
    low = float(np.quantile(values, 0.02))
    high = float(np.quantile(values, 0.98))
    if not np.isfinite(low) or not np.isfinite(high) or math.isclose(low, high):
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    if math.isclose(low, high):
        high = low + 1.0
    return low, high


def model_bounds(model_name: str, fit_vb: bool, fixed_vb: float) -> dict[str, tuple[float, float]]:
    bounds = {
        "K1": (0.0, 10.0),
        "k2": (0.0, 10.0),
        "k3": (0.0, 5.0),
    }
    if model_name == "2tcm-reversible":
        bounds["k4"] = (0.0, 1.0)
    if fit_vb:
        bounds["vB"] = (0.0, 1.0)
    else:
        bounds["vB"] = (fixed_vb, fixed_vb)
    return bounds


def sample_piecewise_linear_with_derivative(time: np.ndarray, signal: np.ndarray, t: float) -> tuple[float, float]:
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
    out = np.zeros_like(aif, dtype=np.float64)
    for i, t in enumerate(time):
        value, deriv = sample_piecewise_linear_with_derivative(time, aif, float(t) + float(delay))
        out[i] = max(0.0, value + tau * deriv)
    return out


def exp_conv_step(z: float, rate: float, dt: float, c0: float, c1: float) -> float:
    if dt <= 0.0:
        return z
    if abs(rate) < 1e-12:
        return z + 0.5 * (c0 + c1) * dt
    decay = math.exp(-rate * dt)
    i0 = (1.0 - decay) / rate
    i1 = (1.0 - decay * (1.0 + rate * dt)) / (rate * rate)
    slope = (c1 - c0) / dt
    return decay * z + c1 * i0 - slope * i1


def predict_irreversible(
    time: np.ndarray,
    aif: np.ndarray,
    K1: float,
    k2: float,
    k3: float,
    vB: float,
    delay: float = 0.0,
    dispersion: float = 0.0,
) -> np.ndarray:
    aif_eff = transform_aif(time, aif, delay, dispersion) if abs(delay) > 1e-12 or dispersion > 1e-12 else aif
    lam = max(k2 + k3, 1e-12)
    z = 0.0
    icp = 0.0
    out = np.zeros_like(time, dtype=np.float64)
    if time[0] > 0:
        z = exp_conv_step(0.0, lam, float(time[0]), 0.0, float(aif_eff[0]))
        icp = 0.5 * float(aif_eff[0]) * float(time[0])
    out[0] = (1.0 - vB) * (K1 / lam * (k3 * icp + k2 * z)) + vB * aif_eff[0]
    for i in range(len(time) - 1):
        dt = float(time[i + 1] - time[i])
        z = exp_conv_step(z, lam, dt, float(aif_eff[i]), float(aif_eff[i + 1]))
        icp += 0.5 * float(aif_eff[i] + aif_eff[i + 1]) * dt
        tissue = K1 / lam * (k3 * icp + k2 * z)
        out[i + 1] = (1.0 - vB) * tissue + vB * aif_eff[i + 1]
    return out


def predict_reversible(
    time: np.ndarray,
    aif: np.ndarray,
    K1: float,
    k2: float,
    k3: float,
    k4: float,
    vB: float,
    delay: float = 0.0,
    dispersion: float = 0.0,
) -> np.ndarray:
    aif_eff = transform_aif(time, aif, delay, dispersion) if abs(delay) > 1e-12 or dispersion > 1e-12 else aif
    rate_sum = k2 + k3 + k4
    discriminant = max(rate_sum * rate_sum - 4.0 * k2 * k4, 0.0)
    root = math.sqrt(discriminant)
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
        z1 = exp_conv_step(0.0, a1, float(time[0]), 0.0, float(aif_eff[0]))
        z2 = exp_conv_step(0.0, a2, float(time[0]), 0.0, float(aif_eff[0]))
    out[0] = (1.0 - vB) * (c1 * z1 + c2 * z2) + vB * aif_eff[0]
    for i in range(len(time) - 1):
        dt = float(time[i + 1] - time[i])
        z1 = exp_conv_step(z1, a1, dt, float(aif_eff[i]), float(aif_eff[i + 1]))
        z2 = exp_conv_step(z2, a2, dt, float(aif_eff[i]), float(aif_eff[i + 1]))
        out[i + 1] = (1.0 - vB) * (c1 * z1 + c2 * z2) + vB * aif_eff[i + 1]
    return out


@dataclass
class Dataset:
    input_dir: Path
    fit_dir: Path
    image: np.ndarray
    time: np.ndarray
    aif: np.ndarray
    maps: dict[str, np.ndarray]
    metadata: dict[str, str]
    bounds: dict[str, tuple[float, float]]

    @property
    def model_name(self) -> str:
        return self.metadata.get("model", "2tcm-reversible")

    @property
    def is_reversible(self) -> bool:
        return self.model_name == "2tcm-reversible"

    @property
    def fit_vb(self) -> bool:
        return self.metadata.get("fit_vb", "1") == "1"

    @property
    def fixed_vb(self) -> float:
        return float(self.metadata.get("fixed_vb", "0.05"))

    @property
    def fit_delay(self) -> bool:
        return self.metadata.get("fit_delay", "0") == "1"

    @property
    def fixed_delay(self) -> float:
        return float(self.metadata.get("fixed_delay", "0.0"))

    @property
    def fit_dispersion(self) -> bool:
        return self.metadata.get("fit_dispersion", "0") == "1"

    @property
    def fixed_dispersion(self) -> float:
        return float(self.metadata.get("fixed_dispersion", "0.0"))

    @property
    def mask(self) -> np.ndarray:
        return self.maps["mask"].astype(bool)

    def predict_tac(self, z: int, y: int, x: int) -> np.ndarray | None:
        if not self.mask[z, y, x]:
            return None
        K1 = float(self.maps["K1"][z, y, x])
        k2 = float(self.maps["k2"][z, y, x])
        k3 = float(self.maps["k3"][z, y, x])
        vB = float(self.maps["vB"][z, y, x])
        delay = float(self.maps["delay"][z, y, x]) if "delay" in self.maps else self.fixed_delay
        dispersion = (
            float(self.maps["dispersion"][z, y, x]) if "dispersion" in self.maps else self.fixed_dispersion
        )
        if not all(np.isfinite([K1, k2, k3, vB, delay, dispersion])):
            return None
        if self.is_reversible:
            k4 = float(self.maps["k4"][z, y, x])
            if not np.isfinite(k4):
                return None
            return predict_reversible(self.time, self.aif, K1, k2, k3, k4, vB, delay, dispersion)
        return predict_irreversible(self.time, self.aif, K1, k2, k3, vB, delay, dispersion)


def load_dataset(input_dir: Path, fit_dir: Path) -> Dataset:
    image = np.load(input_dir / "image.npy")
    time = np.load(input_dir / "time.npy")
    aif = np.load(input_dir / "aif.npy")
    metadata = read_run_metadata(fit_dir / "run.txt")

    if image.ndim != 4:
        raise ValueError("image.npy must have shape (time, depth, height, width)")
    if time.ndim != 1 or aif.ndim != 1:
        raise ValueError("time.npy and aif.npy must be 1D vectors")
    if image.shape[0] != time.shape[0] or time.shape[0] != aif.shape[0]:
        raise ValueError("time dimension mismatch between image, time, and aif")

    maps = {
        "K1": np.load(fit_dir / "K1.npy"),
        "Ki": np.load(fit_dir / "Ki.npy"),
        "k2": np.load(fit_dir / "k2.npy"),
        "k3": np.load(fit_dir / "k3.npy"),
        "mask": np.load(fit_dir / "mask.npy"),
        "rmse": np.load(fit_dir / "rmse.npy"),
        "status": np.load(fit_dir / "status.npy"),
        "iterations": np.load(fit_dir / "iterations.npy"),
        "vB": np.load(fit_dir / "vB.npy"),
        "K1_rawfit": np.load(fit_dir / "K1_rawfit.npy"),
        "Ki_rawfit": np.load(fit_dir / "Ki_rawfit.npy"),
        "k2_rawfit": np.load(fit_dir / "k2_rawfit.npy"),
        "k3_rawfit": np.load(fit_dir / "k3_rawfit.npy"),
        "vB_rawfit": np.load(fit_dir / "vB_rawfit.npy"),
    }
    if (fit_dir / "k4.npy").exists():
        maps["k4"] = np.load(fit_dir / "k4.npy")
    if (fit_dir / "k4_rawfit.npy").exists():
        maps["k4_rawfit"] = np.load(fit_dir / "k4_rawfit.npy")
    if (fit_dir / "delay.npy").exists():
        maps["delay"] = np.load(fit_dir / "delay.npy")
    if (fit_dir / "dispersion.npy").exists():
        maps["dispersion"] = np.load(fit_dir / "dispersion.npy")
    if (fit_dir / "mask_raw.npy").exists():
        maps["mask_raw"] = np.load(fit_dir / "mask_raw.npy")
    if (fit_dir / "mask_removed.npy").exists():
        maps["mask_removed"] = np.load(fit_dir / "mask_removed.npy")
    if (fit_dir / "mask_filled.npy").exists():
        maps["mask_filled"] = np.load(fit_dir / "mask_filled.npy")
    if (fit_dir / "postfilter_target.npy").exists():
        maps["postfilter_target"] = np.load(fit_dir / "postfilter_target.npy")
    if (fit_dir / "postfilter_applied.npy").exists():
        maps["postfilter_applied"] = np.load(fit_dir / "postfilter_applied.npy")
    if (fit_dir / "postfilter_unresolved.npy").exists():
        maps["postfilter_unresolved"] = np.load(fit_dir / "postfilter_unresolved.npy")

    fit_vb = metadata.get("fit_vb", "1") == "1"
    fixed_vb = float(metadata.get("fixed_vb", "0.05"))
    bounds = model_bounds(metadata.get("model", "2tcm-reversible"), fit_vb, fixed_vb)
    if metadata.get("fit_delay", "0") == "1":
        bounds["delay"] = (-0.2, 0.2)
    elif abs(float(metadata.get("fixed_delay", "0.0"))) > 1e-12 and "delay" in maps:
        fixed_delay = float(metadata.get("fixed_delay", "0.0"))
        bounds["delay"] = (fixed_delay, fixed_delay)
    if metadata.get("fit_dispersion", "0") == "1":
        bounds["dispersion"] = (0.0, 0.1)
    elif abs(float(metadata.get("fixed_dispersion", "0.0"))) > 1e-12 and "dispersion" in maps:
        fixed_dispersion = float(metadata.get("fixed_dispersion", "0.0"))
        bounds["dispersion"] = (fixed_dispersion, fixed_dispersion)
    return Dataset(input_dir, fit_dir, image, time, aif, maps, metadata, bounds)


def build_derived_maps(dataset: Dataset) -> dict[str, np.ndarray]:
    image = dataset.image
    mask = dataset.mask
    peak = image.max(axis=0)
    mean = image.mean(axis=0)
    auc = np.trapezoid(image, x=dataset.time, axis=0)
    time_to_peak = dataset.time[np.argmax(image, axis=0)]
    rmse = dataset.maps["rmse"]
    rel_rmse_peak = np.where(mask, rmse / np.maximum(peak, 1e-6), np.nan)
    rel_rmse_auc = np.where(mask, rmse / np.maximum(auc, 1e-6), np.nan)

    lower_hits = np.zeros_like(mask, dtype=np.int32)
    upper_hits = np.zeros_like(mask, dtype=np.int32)
    for name, (lower, upper) in dataset.bounds.items():
        if name not in dataset.maps:
            continue
        arr = dataset.maps[name]
        lower_hits += np.isclose(arr, lower, atol=1e-8)
        upper_hits += np.isclose(arr, upper, atol=1e-8)

    any_bound = ((lower_hits + upper_hits) > 0).astype(np.int32)
    return {
        "peak_image": peak,
        "mean_image": mean,
        "auc_image": auc,
        "time_to_peak": time_to_peak,
        "relative_rmse_peak": rel_rmse_peak,
        "relative_rmse_auc": rel_rmse_auc,
        "lower_bound_hits": lower_hits,
        "upper_bound_hits": upper_hits,
        "total_bound_hits": lower_hits + upper_hits,
        "any_bound_hit": any_bound,
    }


def initial_voxel(dataset: Dataset, voxel: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if voxel is not None:
        return voxel
    peak = dataset.image.max(axis=0)
    masked_peak = np.where(dataset.mask, peak, -np.inf)
    linear = int(np.argmax(masked_peak))
    return tuple(int(v) for v in np.unravel_index(linear, dataset.mask.shape))


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.input_dir, args.fit_dir)
    derived = build_derived_maps(dataset)

    try:
        import napari
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QCheckBox,
            QDoubleSpinBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise SystemExit(
            "Napari visualization dependencies are missing. "
            "Install the project dependencies first, then rerun."
        ) from exc

    viewer = napari.Viewer(title="TCM Parametric Map Viewer")

    try:
        viewer.dims.axis_labels = ("time", "z", "y", "x")
    except Exception:
        pass

    dynamic_limits = robust_limits(dataset.image)
    viewer.add_image(
        dataset.image,
        name="dynamic_pet",
        colormap="gray",
        rendering="mip",
        contrast_limits=dynamic_limits,
        visible=False,
    )

    viewer.add_image(
        derived["peak_image"],
        name="peak_pet",
        colormap="gray",
        opacity=0.9,
        contrast_limits=robust_limits(derived["peak_image"], dataset.mask),
        visible=False,
    )
    viewer.add_image(
        derived["mean_image"],
        name="mean_pet",
        colormap="gray",
        opacity=0.9,
        contrast_limits=robust_limits(derived["mean_image"], dataset.mask),
        visible=False,
    )
    viewer.add_image(
        derived["auc_image"],
        name="auc_pet",
        colormap="magma",
        opacity=0.9,
        contrast_limits=robust_limits(derived["auc_image"], dataset.mask),
        visible=False,
    )
    viewer.add_image(
        derived["time_to_peak"],
        name="time_to_peak",
        colormap="turbo",
        opacity=0.8,
        contrast_limits=robust_limits(derived["time_to_peak"], dataset.mask),
        visible=False,
    )

    for name in ["K1", "k2", "k3", "Ki", "vB", "rmse", "delay", "dispersion", "K1_rawfit", "k2_rawfit", "k3_rawfit", "vB_rawfit", "Ki_rawfit"]:
        if name not in dataset.maps:
            continue
        viewer.add_image(
            dataset.maps[name],
            name=name,
            colormap="viridis" if name not in {"rmse"} else "magma",
            opacity=0.85,
            contrast_limits=robust_limits(dataset.maps[name], dataset.mask),
            visible=name in {"Ki"},
        )

    if "k4" in dataset.maps:
        viewer.add_image(
            dataset.maps["k4"],
            name="k4",
            colormap="viridis",
            opacity=0.85,
            contrast_limits=robust_limits(dataset.maps["k4"], dataset.mask),
            visible=False,
        )

    viewer.add_image(
        derived["relative_rmse_peak"],
        name="relative_rmse_peak",
        colormap="inferno",
        opacity=0.85,
        contrast_limits=robust_limits(derived["relative_rmse_peak"], dataset.mask),
        visible=False,
    )
    viewer.add_image(
        derived["relative_rmse_auc"],
        name="relative_rmse_auc",
        colormap="inferno",
        opacity=0.85,
        contrast_limits=robust_limits(derived["relative_rmse_auc"], dataset.mask),
        visible=False,
    )
    viewer.add_image(
        dataset.maps["iterations"],
        name="iterations",
        colormap="plasma",
        opacity=0.8,
        contrast_limits=robust_limits(dataset.maps["iterations"], dataset.mask),
        visible=False,
    )

    viewer.add_labels(dataset.mask.astype(np.int32), name="mask", opacity=0.35, visible=False)
    if "mask_raw" in dataset.maps:
        viewer.add_labels(dataset.maps["mask_raw"].astype(np.int32), name="mask_raw", opacity=0.25, visible=False)
    if "mask_removed" in dataset.maps:
        viewer.add_labels(
            dataset.maps["mask_removed"].astype(np.int32),
            name="mask_removed",
            opacity=0.6,
            visible=False,
        )
    if "mask_filled" in dataset.maps:
        viewer.add_labels(
            dataset.maps["mask_filled"].astype(np.int32),
            name="mask_filled",
            opacity=0.6,
            visible=False,
        )
    if "postfilter_target" in dataset.maps:
        viewer.add_labels(
            dataset.maps["postfilter_target"].astype(np.int32),
            name="postfilter_target",
            opacity=0.45,
            visible=False,
        )
    if "postfilter_applied" in dataset.maps:
        viewer.add_labels(
            dataset.maps["postfilter_applied"].astype(np.int32),
            name="postfilter_applied",
            opacity=0.55,
            visible=False,
        )
    if "postfilter_unresolved" in dataset.maps:
        viewer.add_labels(
            dataset.maps["postfilter_unresolved"].astype(np.int32),
            name="postfilter_unresolved",
            opacity=0.55,
            visible=False,
        )
    viewer.add_labels(dataset.maps["status"].astype(np.int32), name="status", opacity=0.5, visible=False)
    viewer.add_labels(derived["lower_bound_hits"], name="lower_bound_hits", opacity=0.5, visible=False)
    viewer.add_labels(derived["upper_bound_hits"], name="upper_bound_hits", opacity=0.5, visible=False)
    viewer.add_labels(derived["total_bound_hits"], name="total_bound_hits", opacity=0.5, visible=False)
    viewer.add_labels(derived["any_bound_hit"], name="any_bound_hit", opacity=0.45, visible=False)
    quality_mask_layer = viewer.add_labels(
        np.where(np.isfinite(derived["relative_rmse_auc"]) & dataset.mask, 1, 0).astype(np.int32),
        name="quality_mask_relative_rmse_auc",
        opacity=0.4,
        visible=False,
    )

    selected = initial_voxel(dataset, tuple(args.initial_voxel) if args.initial_voxel else None)
    point_layer = viewer.add_points(
        np.array([selected], dtype=float),
        name="selected_voxel",
        size=2,
        face_color="yellow",
        border_color="black",
        ndim=3,
        visible=False,
    )

    compatible_layers: list = []
    for layer in list(viewer.layers):
        data = getattr(layer, "data", None)
        if layer.name == "selected_voxel" or data is None:
            continue
        if not isinstance(layer, napari.layers.Image):
            continue
        if data.ndim == 3 and data.shape == dataset.mask.shape:
            compatible_layers.append(layer)
        elif data.ndim == 4 and data.shape[1:] == dataset.mask.shape:
            compatible_layers.append(layer)

    class VoxelInspector(QWidget):
        def __init__(self, data: Dataset, marker_layer) -> None:
            super().__init__()
            self.data = data
            self.marker_layer = marker_layer

            layout = QVBoxLayout()
            self.setLayout(layout)

            self.info = QLabel()
            self.info.setWordWrap(True)
            layout.addWidget(self.info)

            self.figure = Figure(figsize=(6, 6))
            self.canvas = FigureCanvasQTAgg(self.figure)
            layout.addWidget(self.canvas)
            self.ax_tac = self.figure.add_subplot(2, 1, 1)
            self.ax_residual = self.figure.add_subplot(2, 1, 2)

            self.update_voxel(*selected)

        def update_voxel(self, z: int, y: int, x: int) -> None:
            shape = self.data.mask.shape
            z = int(np.clip(z, 0, shape[0] - 1))
            y = int(np.clip(y, 0, shape[1] - 1))
            x = int(np.clip(x, 0, shape[2] - 1))
            self.marker_layer.data = np.array([[z, y, x]], dtype=float)

            tac = self.data.image[:, z, y, x]
            pred = self.data.predict_tac(z, y, x)
            rmse = float(self.data.maps["rmse"][z, y, x])
            peak = float(tac.max())
            rel_peak = rmse / max(peak, 1e-6)
            status = int(self.data.maps["status"][z, y, x])
            iterations = int(self.data.maps["iterations"][z, y, x])
            in_mask = bool(self.data.mask[z, y, x])

            parameter_text = [
                f"K1={float(self.data.maps['K1'][z, y, x]):.4g}",
                f"k2={float(self.data.maps['k2'][z, y, x]):.4g}",
                f"k3={float(self.data.maps['k3'][z, y, x]):.4g}",
            ]
            if "k4" in self.data.maps:
                parameter_text.append(f"k4={float(self.data.maps['k4'][z, y, x]):.4g}")
            parameter_text.append(f"vB={float(self.data.maps['vB'][z, y, x]):.4g}")
            if "delay" in self.data.maps:
                parameter_text.append(f"delay={float(self.data.maps['delay'][z, y, x]):.4g}")
            if "dispersion" in self.data.maps:
                parameter_text.append(f"dispersion={float(self.data.maps['dispersion'][z, y, x]):.4g}")
            parameter_text.append(f"Ki={float(self.data.maps['Ki'][z, y, x]):.4g}")

            self.info.setText(
                "\n".join(
                    [
                        f"voxel = ({z}, {y}, {x})",
                        f"in_mask = {int(in_mask)}",
                        f"rmse = {rmse:.4g}",
                        f"relative_rmse_peak = {rel_peak:.4g}",
                        f"status = {status}",
                        f"iterations = {iterations}",
                        ", ".join(parameter_text),
                    ]
                )
            )

            self.ax_tac.clear()
            self.ax_residual.clear()

            scaled_aif = self.data.aif / max(float(np.max(self.data.aif)), 1e-6) * max(peak, 1e-6)
            self.ax_tac.plot(self.data.time, tac, "o-", label="measured TAC", linewidth=2)
            self.ax_tac.plot(self.data.time, scaled_aif, "--", label="scaled AIF", alpha=0.5)
            if pred is not None:
                self.ax_tac.plot(self.data.time, pred, "-", label="model fit", linewidth=2)
                self.ax_residual.axhline(0.0, color="black", linewidth=1)
                self.ax_residual.plot(self.data.time, pred - tac, "o-", color="tab:red")
            else:
                self.ax_residual.axhline(0.0, color="black", linewidth=1)
                self.ax_residual.text(
                    0.5,
                    0.5,
                    "No valid fit for this voxel",
                    transform=self.ax_residual.transAxes,
                    ha="center",
                    va="center",
                )

            self.ax_tac.set_ylabel("activity")
            self.ax_tac.set_title("Measured TAC and fitted curve")
            self.ax_tac.legend(loc="best")
            self.ax_residual.set_title("Residuals: fit - measured")
            self.ax_residual.set_xlabel("time [min]")
            self.ax_residual.set_ylabel("residual")
            self.figure.tight_layout()
            self.canvas.draw_idle()

    class QualityMaskController(QWidget):
        def __init__(
            self,
            viewer_instance,
            metric_map: np.ndarray,
            spatial_mask: np.ndarray,
            layers: list,
            pass_layer,
        ) -> None:
            super().__init__()
            self.viewer = viewer_instance
            self.metric_map = metric_map
            self.spatial_mask = spatial_mask.astype(bool)
            self.layers = layers
            self.pass_layer = pass_layer
            self.original_data = {id(layer): np.array(layer.data, copy=True) for layer in layers}
            self.current_layer = None

            finite_metric = metric_map[np.isfinite(metric_map) & self.spatial_mask]
            self.metric_min = float(np.nanmin(finite_metric)) if finite_metric.size else 0.0
            self.metric_max = float(np.nanmax(finite_metric)) if finite_metric.size else 1.0
            if math.isclose(self.metric_min, self.metric_max):
                self.metric_max = self.metric_min + 1.0
            self.slider_steps = 1000

            layout = QVBoxLayout()
            self.setLayout(layout)

            title = QLabel("Quality Mask: relative_rmse_auc")
            layout.addWidget(title)

            info = QLabel(
                "Select a scalar image layer, then threshold it so only voxels with "
                "relative_rmse_auc less than or equal to the chosen value remain visible."
            )
            info.setWordWrap(True)
            layout.addWidget(info)

            self.enabled = QCheckBox("Enable threshold on selected layer")
            self.enabled.setChecked(True)
            layout.addWidget(self.enabled)

            self.selected_label = QLabel()
            self.selected_label.setWordWrap(True)
            layout.addWidget(self.selected_label)

            row = QHBoxLayout()
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, self.slider_steps)
            self.slider.setValue(self.slider_steps)
            row.addWidget(self.slider)

            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(4)
            self.spinbox.setRange(self.metric_min, self.metric_max)
            self.spinbox.setSingleStep(max((self.metric_max - self.metric_min) / 100.0, 1e-4))
            self.spinbox.setValue(self.metric_max)
            row.addWidget(self.spinbox)
            layout.addLayout(row)

            self.summary_label = QLabel()
            self.summary_label.setWordWrap(True)
            layout.addWidget(self.summary_label)

            reset_button = QPushButton("Reset Threshold")
            layout.addWidget(reset_button)

            self.slider.valueChanged.connect(self._on_slider_changed)
            self.spinbox.valueChanged.connect(self._on_spinbox_changed)
            self.enabled.stateChanged.connect(self._on_enabled_changed)
            reset_button.clicked.connect(self._reset_threshold)
            self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

            self._sync_summary()
            self._on_active_layer_changed()

        def _is_compatible(self, layer) -> bool:
            return layer in self.layers

        def _metric_threshold(self) -> float:
            return float(self.spinbox.value())

        def _quality_pass_mask(self) -> np.ndarray:
            threshold = self._metric_threshold()
            return self.spatial_mask & np.isfinite(self.metric_map) & (self.metric_map <= threshold)

        def _sync_summary(self) -> None:
            pass_mask = self._quality_pass_mask()
            kept = int(pass_mask.sum())
            total = int(self.spatial_mask.sum())
            fraction = kept / max(total, 1)
            self.pass_layer.data = pass_mask.astype(np.int32)
            self.summary_label.setText(
                f"Threshold = {self._metric_threshold():.4f}\n"
                f"Retained voxels = {kept} / {total} ({fraction:.1%})"
            )

        def _slider_to_value(self, position: int) -> float:
            alpha = position / self.slider_steps
            return self.metric_min + alpha * (self.metric_max - self.metric_min)

        def _value_to_slider(self, value: float) -> int:
            alpha = (value - self.metric_min) / max(self.metric_max - self.metric_min, 1e-12)
            return int(round(np.clip(alpha, 0.0, 1.0) * self.slider_steps))

        def _restore_layer(self, layer) -> None:
            if layer is None or not self._is_compatible(layer):
                return
            contrast = getattr(layer, "contrast_limits", None)
            layer.data = np.array(self.original_data[id(layer)], copy=True)
            if contrast is not None:
                layer.contrast_limits = contrast

        def _apply_to_layer(self, layer) -> None:
            if layer is None or not self._is_compatible(layer):
                return
            contrast = getattr(layer, "contrast_limits", None)
            original = self.original_data[id(layer)]
            if not self.enabled.isChecked():
                layer.data = np.array(original, copy=True)
            else:
                keep = self._quality_pass_mask()
                if original.ndim == 4:
                    masked = np.where(keep[None, ...], original, np.nan)
                else:
                    masked = np.where(keep, original, np.nan)
                layer.data = masked
            if contrast is not None:
                layer.contrast_limits = contrast

        def _set_current_layer(self, layer) -> None:
            if layer is self.current_layer:
                self._update_selected_label()
                self._apply_to_layer(layer)
                return
            self._restore_layer(self.current_layer)
            self.current_layer = layer if self._is_compatible(layer) else None
            self._update_selected_label()
            self._apply_to_layer(self.current_layer)

        def _update_selected_label(self) -> None:
            if self.current_layer is None:
                self.selected_label.setText("Selected layer: none or not threshold-compatible")
            else:
                self.selected_label.setText(f"Selected layer: {self.current_layer.name}")

        def _on_active_layer_changed(self, event=None) -> None:
            active = self.viewer.layers.selection.active
            self._set_current_layer(active)

        def _on_slider_changed(self, position: int) -> None:
            value = self._slider_to_value(position)
            self.spinbox.blockSignals(True)
            self.spinbox.setValue(value)
            self.spinbox.blockSignals(False)
            self._sync_summary()
            self._apply_to_layer(self.current_layer)

        def _on_spinbox_changed(self, value: float) -> None:
            position = self._value_to_slider(value)
            self.slider.blockSignals(True)
            self.slider.setValue(position)
            self.slider.blockSignals(False)
            self._sync_summary()
            self._apply_to_layer(self.current_layer)

        def _on_enabled_changed(self, state: int) -> None:
            self._sync_summary()
            self._apply_to_layer(self.current_layer)

        def _reset_threshold(self) -> None:
            self.spinbox.setValue(self.metric_max)
            self.enabled.setChecked(True)

    inspector = VoxelInspector(dataset, point_layer)
    viewer.window.add_dock_widget(inspector, area="right", name="voxel inspector")
    quality_controller = QualityMaskController(
        viewer,
        derived["relative_rmse_auc"],
        dataset.mask,
        compatible_layers,
        quality_mask_layer,
    )
    viewer.window.add_dock_widget(quality_controller, area="right", name="quality mask")

    def on_click(layer, event) -> None:
        data_pos = layer.world_to_data(event.position)
        coords = [int(round(v)) for v in data_pos]
        if len(coords) >= 4:
            _, z, y, x = coords[-4:]
        elif len(coords) >= 3:
            z, y, x = coords[-3:]
        else:
            return
        inspector.update_voxel(z, y, x)

    for layer in list(viewer.layers):
        layer.mouse_drag_callbacks.append(on_click)

    napari.run()


if __name__ == "__main__":
    main()
