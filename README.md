# Tissue Compartment Model TRF Solver

Standalone C++ solver for voxel-wise nonlinear fitting of the irreversible and reversible two-tissue compartment models for dynamic PET.

The implementation is built around three decisions:

1. Use a specialized small-problem trust-region Gauss-Newton solver instead of a large general optimization library.
2. Use your existing linear least-squares formulation as a warm start before nonlinear refinement.
3. Parallelize across voxels with OpenMP, which is the natural first level of parallelism for voxel-wise TAC fitting.

## What Is Implemented

- Irreversible 2TCM (`K1`, `k2`, `k3`, optional `vB`)
- Reversible 2TCM (`K1`, `k2`, `k3`, `k4`, optional `vB`)
- Bounded optimization through parameter transforms into a trust-region Gauss-Newton loop
- Linear warm starts derived from the same basis-transform idea already present in your adjacent Torch code
- Minimal `.npy` input/output for easy interop with NumPy
- Synthetic smoke test covering both model variants

## Input Format

The C++ executable reads three `.npy` files from an input directory:

- `image.npy`: shape `(time, depth, height, width)`
- `time.npy`: shape `(time,)`, in minutes
- `aif.npy`: shape `(time,)`

The helper script [scripts/unpack_npz.py](/Users/csa047/uit/phd/projects/tissue_compartment_model_trf/scripts/unpack_npz.py) converts a packed `.npz` bundle into that layout.

Example:

```bash
python3 scripts/unpack_npz.py sample.npz input_dir --aif-key aif
```

## Build

```bash
cmake -S . -B build
cmake --build build
```

If OpenMP is available, CMake enables it automatically.

## Run

Reversible 2TCM with fitted blood volume:

```bash
./build/tcm_fit \
  --input-dir input_dir \
  --output-dir output_rev \
  --model rev \
  --fit-vb 1 \
  --mask-threshold 0.05 \
  --mask-keep-largest-component 1 \
  --mask-fill-holes 1
```

Irreversible 2TCM with fixed blood volume:

```bash
./build/tcm_fit \
  --input-dir input_dir \
  --output-dir output_irr \
  --model irr \
  --fit-vb 0 \
  --fixed-vb 0.05 \
  --mask-threshold 0.05
```

## Outputs

The solver writes voxel-wise maps as `.npy` arrays with shape `(depth, height, width)`:

- `K1.npy`
- `k2.npy`
- `k3.npy`
- `k4.npy` for the reversible model
- `vB.npy`
- `Ki.npy`
- `rmse.npy`
- `iterations.npy`
- `status.npy`
- `mask.npy`
- `mask_raw.npy`
- `mask_removed.npy`
- `mask_filled.npy`

`status.npy` uses:

- `1`: gradient convergence
- `2`: step convergence
- `3`: cost convergence
- `4`: hit iteration limit
- `5`: invalid input

## Validation

The included smoke test generates noiseless synthetic data, builds the solver, runs both models, and checks parameter recovery:

```bash
python3 tests/smoke_test.py
```

On the local run used during development, the smoke test recovered parameters with mean absolute errors on the order of `1e-6` to `1e-3`, depending on model and parameter.

## Why This Design

The main bottleneck in voxel-wise PET fitting is not a single hard optimization problem. It is a very large number of tiny optimization problems. That changes what is worth optimizing.

- A specialized forward model is cheaper than calling a general library millions of times.
- Hard bounds matter because unbounded LM can walk into non-physical regions and become unstable.
- A linear warm start is worth keeping even if it is biased under noise, because it gives a fast, physically plausible initialization for the nonlinear solver.
- OpenMP is the best first parallel layer because the workload is embarrassingly parallel across voxels. MPI should come next as coarse partitioning across subjects or voxel blocks.

## Current Limitations

- The current forward model predicts point samples at the provided times. If your PET frames represent frame averages, the next upgrade should be to support frame start/end times and integrate over each frame.
- The solver uses trust-region Gauss-Newton with bounded parameter transforms rather than a line-by-line reimplementation of SciPy's exact reflective TRF internals.
- MPI is not wired in yet. The code is structured so that the next step is to split the flattened voxel range across ranks and let each rank write partial outputs.
- Only `.npy` I/O is implemented in C++. Packed `.npz` input is handled through the Python preprocessing helper.
- Mask cleanup currently assumes the body should be a single dominant 3D connected component. That is appropriate for your current whole-subject use case, but it may need adjustment for scans with truly disconnected anatomy in the field of view.

## Recommended Next Steps

1. Validate against your existing `curve_fit(..., method="trf")` implementation on a representative subset of voxels.
2. Add frame-average forward modeling if your `time` vector stores frame midpoints.
3. Add voxel-wise restart logic for bad fits, for example retrying from a conservative default or from multiple warm starts only when the first fit is poor.
4. Add MPI over voxel blocks or subject batches once you move from workstation-scale to cluster-scale runs.
