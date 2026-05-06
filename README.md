# KinePET

This repo now ships one fitting core:

- a TAC-based bounded nonlinear solver for dynamic PET two-tissue compartment models

Supported models:

- `irr`: irreversible 2TCM
- `rev`: reversible 2TCM

The solver is available through:

- a standalone C++ CLI: `tcm_fit_tacs`
- a Python package with a nanobind extension: `kinepet`

The old image-oriented fitting pipeline has been removed on purpose. Image preprocessing, masking, postprocessing, and map reconstruction are expected to live in Python outside the core solver.

## What is in the repo

- `src/solver.cpp`, `src/solver.hpp`
  - bounded trust-region / LM-style nonlinear solver
- `src/tcm_model.cpp`, `src/tcm_model.hpp`
  - model equations, bounds, parameter packing, `Ki`
- `src/lls.cpp`, `src/lls.hpp`
  - linear least-squares warm start
- `src/tac_batch.cpp`, `src/tac_batch.hpp`
  - reusable TAC-batch fitting core shared by the CLI and Python binding
- `src/tac_fit_main.cpp`
  - TAC-batch CLI
- `src/python_bindings.cpp`
  - nanobind extension
- `kinepet/__init__.py`
  - user-facing Python API
- `scripts/make_curves_from_csv.py`
  - convert simulated CSV tables into TAC-batch directories
- `scripts/make_synthetic_dataset.py`
  - generate synthetic TAC batches for testing
- `scripts/extract_voi_reference_fits.py`
  - compare VOI TAC fits across `cpp`, `pypet`, and `lls`
- `tests/test_bindings.py`
  - pytest coverage for the Python binding

## Input format

The solver works on flattened TAC batches.

Required files:

- `tacs.npy`: shape `(T, N)`
- `time.npy`: shape `(T,)`
- `aif.npy`: shape `(T,)`, `(T, 1)`, or `(T, N)`

Interpretation of `aif.npy`:

- `(T,)` or `(T, 1)`: one shared AIF for all TACs
- `(T, N)`: one AIF per TAC

Optional:

- `weights.npy`: shape `(T,)`, `(T, 1)`, or `(T, N)`

Time is interpreted as minutes. If the maximum time value is greater than `60`, the solver assumes the input is in seconds and converts to minutes internally.

## Build and install

For local development:

```bash
uv sync --group dev
```

This installs:

- the Python package in editable mode
- `pytest`
- `nanobind`
- `scikit-build-core`

To build the standalone CLI and the extension through CMake:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)')
cmake --build build
```

If you only need the Python package, `uv sync --group dev` is enough.

## Python API

```python
import numpy as np
import kinepet

fit = kinepet.fit_tacs(
    tacs,
    time,
    aif,
    model="rev",
    fit_vb=True,
    threads=8,
)

print(fit.K1.shape)      # (N,)
print(fit.status.shape)  # (N,)
```

Single-TAC fitting:

```python
fit = kinepet.fit_one_tac(
    tac,
    time,
    aif,
    model="irr",
    fit_vb=True,
)
```

Forward evaluation:

```python
pred = kinepet.evaluate_model(
    time,
    aif,
    model="rev",
    K1=0.12,
    k2=0.16,
    k3=0.09,
    k4=0.04,
    vB=0.05,
)
```

Useful helpers:

- `kinepet.default_bounds(...)`
- `kinepet.default_frame_weights(time)`

## CLI

Build first, then run:

```bash
./build/tcm_fit_tacs --input-dir INPUT --output-dir OUTPUT --model rev|irr [options]
```

Supported options:

- `--fit-vb 0|1`
- `--fixed-vb VALUE`
- `--fit-delay 0|1`
- `--fixed-delay VALUE`
- `--fit-dispersion 0|1`
- `--fixed-dispersion VALUE`
- `--threads N`
- `--max-iter N`
- `--weights-file PATH`
- `--no-linear-warm-start`

Outputs are flat `(N,)` arrays:

- `K1.npy`
- `k2.npy`
- `k3.npy`
- `k4.npy` for `rev`
- `vB.npy`
- `delay.npy` when fitted or fixed to a nonzero value
- `dispersion.npy` when fitted or fixed to a nonzero value
- `Ki.npy`
- `rmse.npy`
- `iterations.npy`
- `status.npy`
- `run.txt`

## Current default bounds

The source of truth is `src/tcm_model.cpp`.

Current defaults:

- `K1`: `[0, 10]`
- `k2`: `[0, 10]`
- `k3`: `[0, 5]`
- `k4`: `[0, 1]` when active
- `vB`: `[0, 1]` when fitted
- `delay`: `[-0.2, 0.2]` minutes when fitted
- `dispersion`: `[0, 0.1]` minutes when fitted

## Scripts

Convert simulated CSV tables to TAC batches:

```bash
python3 scripts/make_curves_from_csv.py \
  --csv-glob 'data/simulated-csv/*.csv' \
  --output-root data/simulated-tac-batches \
  --overwrite
```

Generate a synthetic TAC batch:

```bash
python3 scripts/make_synthetic_dataset.py data/synthetic_tacs --model rev --num-tacs 64
```

Run VOI comparison export:

```bash
uv run python scripts/extract_voi_reference_fits.py --threads 1
```

That script uses:

- `time` and `aif` from the raw NPZ files
- `Brain` and `Myocardium` from `VOI_A`
- `kinepet` for the TAC-based solver path
- the PyPET reference implementation
- the installed `lls` package

## Tests

Run:

```bash
uv run pytest -q
```

Current pytest coverage checks:

- irreversible batch fitting with per-TAC AIFs
- reversible single-TAC fitting through the Python binding
- optional helper APIs like `default_bounds`
- input validation on invalid AIF shapes

## Notes

- The nonlinear solver is custom C++, not SciPy TRF copied line-for-line.
- The model is evaluated on the native non-uniform time grid with piecewise-linear AIF handling.
- Delay and dispersion support are available, but simultaneous estimation is still less stable than the core kinetic parameters on difficult curves.
