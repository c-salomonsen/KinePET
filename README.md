# Tissue Compartment Model TRF Solver

Standalone C++ solver and Python tooling for voxel-wise kinetic modeling of dynamic PET with irreversible and reversible two-tissue compartment models.

The current repo has three main pieces:

- a custom C++ voxel-wise nonlinear fitter with OpenMP parallelism
- Python helpers for unpacking raw `.npz` data and batch-running the solver
- analysis/validation tools for comparing the C++ fits against PyPET and `lls`

## What is implemented

- Irreversible 2TCM (`irr`)
- Reversible 2TCM (`rev`)
- Optional fitted or fixed blood volume fraction `vB`
- Optional fitted or fixed AIF delay `Delta`
- Optional fitted or fixed AIF dispersion `tau`
- Bounded nonlinear optimization in C++
- Linear least-squares warm start before nonlinear refinement
- Optional prefit spatial Gaussian smoothing of each 3D frame
- Optional prefit temporal median or Gaussian filtering of each voxel TAC
- Threshold-based 3D masking with optional largest-component cleanup and hole filling
- Selective masked 3D median post-filter for unstable voxels
- Minimal `.npy` input/output for easy NumPy interop
- Napari-based fit visualization
- VOI-level comparison export across `cpp`, `pypet`, and `lls`

## Repo layout

- `src/`: core C++ solver, forward models, mask cleanup, post-filter, `.npy` I/O
- `scripts/unpack_npz.py`: convert one packed subject from `.npz` to solver-ready `.npy`
- `scripts/batch_fit_npz.py`: sequential batch runner over many subjects
- `scripts/view_param_maps_napari.py`: interactive viewer for fitted maps and diagnostics
- `scripts/extract_voi_reference_fits.py`: VOI-level comparison export
- `scripts/build_error_analysis_notebook.py`: populate the error-analysis notebook
- `tests/smoke_test.py`: synthetic regression check

## Input format

The C++ solver reads three files from an input directory:

- `image.npy`: shape `(T, D, H, W)`
- `time.npy`: shape `(T,)`
- `aif.npy`: shape `(T,)`

The code treats time as minutes. If values look like seconds (`max(time) > 60`), both the C++ solver and the VOI export path normalize to minutes automatically.

## Unpack one packed subject

```bash
python3 scripts/unpack_npz.py sample.npz input_dir
```

Default keys expected in the NPZ bundle:

- `image`
- `time`
- `aif`
- optional `sample_id`
- optional `group`

The unpacked directory contains:

- `image.npy`
- `time.npy`
- `aif.npy`
- optional `metadata.json`

## Build

```bash
cmake -S . -B build
cmake --build build
```

If OpenMP is available, CMake enables it automatically. On Apple Clang you may need an explicit OpenMP toolchain; on GCC/Intel HPC toolchains it is usually picked up directly.

## Main CLI

```bash
./build/tcm_fit --input-dir INPUT --output-dir OUTPUT --model rev|irr [options]
```

Main options:

- `--fit-vb 0|1`
- `--fixed-vb VALUE`
- `--fit-delay 0|1`
- `--fixed-delay VALUE`
- `--fit-dispersion 0|1`
- `--fixed-dispersion VALUE`
- `--mask-threshold VALUE`
- `--mask-keep-largest-component 0|1`
- `--mask-fill-holes 0|1`
- `--spatial-gaussian-prefilter 0|1`
- `--spatial-gaussian-radius N`
- `--spatial-gaussian-sigma VALUE`
- `--temporal-filter none|median|gaussian`
- `--temporal-filter-radius N`
- `--temporal-gaussian-sigma VALUE`
- `--postfilter-status-median 0|1`
- `--postfilter-radius N`
- `--postfilter-min-reliable-neighbors N`
- `--threads N`
- `--max-iter N`
- `--weights-file PATH`
- `--no-linear-warm-start`

Reversible fit with fitted `vB`:

```bash
./build/tcm_fit \
  --input-dir data/unpacked/AA1 \
  --output-dir data/fits_rev/AA1 \
  --model rev \
  --fit-vb 1 \
  --mask-threshold 0.2 \
  --mask-keep-largest-component 1 \
  --mask-fill-holes 1 \
  --postfilter-status-median 1 \
  --postfilter-radius 1 \
  --postfilter-min-reliable-neighbors 3 \
  --threads 8
```

Irreversible fit with fixed `vB`:

```bash
./build/tcm_fit \
  --input-dir data/unpacked/AA1 \
  --output-dir data/fits_irr/AA1 \
  --model irr \
  --fit-vb 0 \
  --fixed-vb 0.05 \
  --mask-threshold 0.2 \
  --threads 8
```

## Current default bounds

The active parameter bounds currently come from `src/tcm_model.cpp`.

Base bounds:

- `K1`: `[0, 10]`
- `k2`: `[0, 10]`
- `k3`: `[0, 5]`
- `k4`: `[0, 1]` when used
- `vB`: `[0, 1]` when fitted
- `delay`: `[-0.2, 0.2]` minutes when fitted
- `dispersion`: `[0, 0.1]` minutes when fitted

When `--fit-vb 0`, `vB` is fixed to `--fixed-vb`.
When `--fit-delay 0`, delay is fixed to `--fixed-delay`.
When `--fit-dispersion 0`, dispersion is fixed to `--fixed-dispersion`.

If you change the C++ bounds, also update the Napari viewer logic in `scripts/view_param_maps_napari.py`, which derives bound-hit overlays from a Python copy of the bound assumptions.

## Weighting and warm start

If `--weights-file` is not given, the solver uses default frame weights derived from the time grid.

By default, the nonlinear solver also uses the internal linear least-squares approximation as a warm start. You can disable that with:

```bash
--no-linear-warm-start
```

For delay/dispersion fits, the warm start still initializes the kinetic parameters from the linear model, but delay and dispersion start from conservative defaults. In practice, fitting delay alone or dispersion alone is more stable than fitting both simultaneously.

## Optional prefit smoothing

The repo now supports two optional preprocessing steps that are applied only to the TACs used for fitting. Masking still uses the original image.

Spatial:

- `--spatial-gaussian-prefilter 0|1`
- `--spatial-gaussian-radius N`
- `--spatial-gaussian-sigma VALUE`

This applies a separable 3D Gaussian filter to each dynamic frame before fitting. The original motivation is to reduce isolated voxel-level spikes that create implausible parameter-map speckles.

Temporal:

- `--temporal-filter none|median|gaussian`
- `--temporal-filter-radius N`
- `--temporal-gaussian-sigma VALUE`

This applies a 1D filter along the time axis independently at each voxel before fitting.

Practical guidance:

- spatial Gaussian is the safer first thing to try for isolated map speckles
- temporal median is good for occasional frame outliers
- temporal Gaussian is smoother but more likely to blur the fast early peak

The solver still reports `rmse.npy` against the original raw TACs, not the filtered TACs, so fit diagnostics remain interpretable across filtered and unfiltered runs.

## Outputs

The solver writes `.npy` volumes with shape `(D, H, W)`.

Main fitted maps:

- `K1.npy`
- `k2.npy`
- `k3.npy`
- `k4.npy` for reversible fits
- `vB.npy`
- `delay.npy` when fitted or fixed to a nonzero value
- `dispersion.npy` when fitted or fixed to a nonzero value
- `Ki.npy`
- `rmse.npy`
- `iterations.npy`
- `status.npy`

Raw nonlinear fit before post-filtering:

- `K1_rawfit.npy`
- `k2_rawfit.npy`
- `k3_rawfit.npy`
- `k4_rawfit.npy` for reversible fits
- `vB_rawfit.npy`
- `delay_rawfit.npy` when written
- `dispersion_rawfit.npy` when written
- `Ki_rawfit.npy`

Mask and post-filter diagnostics:

- `mask.npy`
- `mask_raw.npy`
- `mask_removed.npy`
- `mask_filled.npy`
- `postfilter_target.npy`
- `postfilter_applied.npy`
- `postfilter_unresolved.npy`

Run metadata:

- `run.txt`

If post-filtering is enabled, the main parameter maps are the post-filtered versions. The `*_rawfit.npy` files preserve the direct voxel-wise nonlinear result.

## `status.npy` semantics

Internal fit statuses are defined in `src/solver.hpp`, but the saved `status.npy` has an important practical detail: voxels outside the mask and some unwritten error cases remain `0`.

In practice:

- `0`: outside mask, failed fit, or other unwritten/non-success case
- `1`: gradient convergence
- `2`: step convergence
- `3`: cost convergence
- `4`: hit iteration limit

`status = 2` means the proposed update became very small. That usually means the solver stalled on a flat region or near a bound, not necessarily that the voxel is a high-confidence fit.

## Mask cleanup

Masking starts from a simple TAC threshold:

- include voxel if `max(TAC) >= mask_threshold`

Optional cleanup then:

- keeps only the largest connected 3D component
- fills enclosed holes

These steps are controlled by:

- `--mask-keep-largest-component`
- `--mask-fill-holes`

This is aimed at whole-subject small-animal PET volumes where random speckles outside the body should be removed.

## Selective median post-filter

The post-filter is intentionally narrow. It is a stabilization aid, not a substitute for model identifiability.

Rules:

- only voxels inside `mask.npy` are considered
- only voxels with `status` `2` or `4` are targets
- only voxels with `status` `1` or `3` can donate values
- default neighborhood radius is `1`, meaning a masked `3x3x3` window
- at least `3` reliable donor voxels are required
- `K1`, `k2`, `k3`, `k4` when present, and `vB` are filtered independently
- `Ki` is recomputed from the filtered kinetic parameters instead of filtered directly

The diagnostic masks mean:

- `postfilter_target.npy`: voxels marked for possible replacement
- `postfilter_applied.npy`: target voxels that were actually replaced
- `postfilter_unresolved.npy`: targets that did not have enough reliable neighbors

If delay and/or dispersion maps are present, the selective post-filter median-stabilizes those maps too for the targeted voxels.

## Batch pipeline

A typical layout is:

- `data/raw/<sample>.npz`
- `data/unpacked/<sample_id>/`
- `data/fits_rev/<sample_id>/` or `data/fits_irr/<sample_id>/`

Run the sequential batch pipeline like this:

```bash
python3 scripts/batch_fit_npz.py \
  --npz-glob 'data/raw/*.npz' \
  --unpacked-root data/unpacked \
  --fit-root data/fits_rev \
  --tcm-fit build/tcm_fit \
  --model rev \
  --fit-vb 1 \
  --mask-threshold 0.2 \
  --threads 8 \
  --skip-existing
```

Useful batch options:

- `--delete-unpacked-after-fit`
- `--no-linear-warm-start`
- `--fit-delay`, `--fixed-delay`
- `--fit-dispersion`, `--fixed-dispersion`
- `--spatial-gaussian-prefilter`, `--spatial-gaussian-radius`, `--spatial-gaussian-sigma`
- `--temporal-filter`, `--temporal-filter-radius`, `--temporal-gaussian-sigma`
- all major solver flags from the main CLI

The batch runner writes:

- one unpacked directory per sample
- one fit directory per sample
- `batch_manifest.csv` in the fit root

## Napari viewer

Open one fitted subject with:

```bash
uv run python scripts/view_param_maps_napari.py \
  --input-dir data/unpacked/AA1 \
  --fit-dir data/fits_rev/AA1
```

Optional initial voxel:

```bash
--initial-voxel 40 24 24
```

The viewer shows:

- the 4D dynamic PET volume
- summary images such as peak, mean, AUC, and time-to-peak
- fitted parameter maps
- optional `delay` and `dispersion` maps
- fit diagnostics like RMSE and iteration count
- mask and post-filter overlays
- bound-hit overlays derived from the current parameter bounds
- a voxel inspector with measured TAC, predicted TAC, scaled AIF, and residuals

It also supports thresholding the currently selected image layer using `relative_rmse_auc`, which is useful for hiding noisy background voxels.

## Synthetic validation

Run the smoke test:

```bash
python3 tests/smoke_test.py
```

This test:

- regenerates synthetic reversible and irreversible datasets
- regenerates dedicated synthetic delay-only and dispersion-only datasets
- rebuilds the solver
- runs both model variants
- checks mean absolute error for a subset of recovered parameters
- checks delay recovery in a delay-only fit
- checks dispersion recovery in a dispersion-only fit
- checks that simultaneous delay+dispersion output maps are produced and finite

It is the minimum regression check after solver changes.

## VOI-level comparison export

The repo includes a comparison pipeline for expert-defined ROI averages stored in VOI pickle files.

Current script:

```bash
.venv/bin/python scripts/extract_voi_reference_fits.py --threads 1
```

What it does:

- matches raw NPZ subjects to VOI pickle files by sample id
- takes `time` and `aif` from the raw NPZ
- takes `Brain` and `Myocardium` from `loaded["VOI_A"]`
- checks that `VOI_t` matches the NPZ time vector
- exports fits from:
  - the C++ solver (`method=cpp`)
  - the reference PyPET implementation (`method=pypet`)
  - the installed `lls` package (`method=lls`)

Outputs:

- `output/validation/voi_reference/voi_reference_fits.csv`
- `output/validation/voi_reference/voi_reference_curves.npz`
- `output/validation/voi_reference/voi_reference_skipped.csv`
- `output/validation/voi_reference/metadata.json`

Important: the exported `error_value` is not numerically comparable across methods without checking `error_kind`:

- `cpp`: `rmse_native_grid`
- `pypet`: `rss_interp_grid`
- `lls`: `mse_interp_grid`

## Why this design

The core optimization problem is not one hard fit. It is a very large number of tiny fits. That changes what matters:

- a custom small-problem solver is cheaper than calling a large generic optimizer millions of times
- hard bounds matter because unbounded LM can drift into non-physical regions
- the linear solver is still useful even when biased under noise because it gives a fast warm start
- OpenMP is the natural first parallel layer because the work is embarrassingly parallel across voxels

## Current limitations

- The forward model predicts point samples at the provided times. If your PET data are frame averages, frame-integrated modeling is the next important upgrade.
- The nonlinear solver is bounded and trust-region-like, but it is not a line-by-line reimplementation of SciPy’s reflective TRF internals.
- MPI is not wired into the codebase yet.
- Packed `.npz` I/O is handled in Python, not in C++.
- The post-filter improves map stability but should not be mistaken for a cure for true identifiability problems.

## Recommended workflow after code changes

If you change solver logic, output files, bounds, or status semantics:

1. run `python3 tests/smoke_test.py`
2. fit at least one real subject
3. open that subject in Napari if any viewer-facing output changed
4. rerun `scripts/extract_voi_reference_fits.py` if comparison/export semantics changed

## Next likely upgrades

1. Frame-average forward modeling.
2. Better regularization or restart logic for simultaneous delay+dispersion fits.
3. MPI over subjects or voxel blocks.
4. Smarter restart logic for poor fits.
5. More systematic comparison notebooks for `cpp` vs `pypet` vs `lls`.
