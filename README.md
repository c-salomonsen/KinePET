# KinePET

This repo now ships one fitting core:

- a TAC-based bounded nonlinear solver for dynamic PET two-tissue compartment models

Supported models:

- `irr`: irreversible 2TCM
- `rev`: reversible 2TCM

The solver is available through:

- a standalone C++ CLI: `tcm_fit_tacs`
- a Python package with a nanobind extension: `kinepet`

Supported Python versions for the published package:

- `3.11+`

The old image-oriented fitting pipeline has been removed on purpose. Image preprocessing, masking, postprocessing, and map reconstruction are expected to live in Python outside the core solver.

## What is in the repo

The fitting path is **Python/CLI → batch orchestration → per-TAC solver → model evaluation**.

| Responsibility | Location |
| --- | --- |
| Kinetic model types and fitted/fixed blood fraction | `src/model.hpp` |
| Irreversible / reversible tissue equations | `src/models/irreversible.cpp`, `src/models/reversible.cpp` |
| Shared AIF transformation and vascular mixing | `src/aif.cpp`, `src/tcm_model.cpp` |
| Exact piecewise-linear exponential convolution | `src/models/convolution.cpp` |
| Parameter ordering, bounds, and defaults | `src/parameters.cpp` |
| LLS dispatch and model-specific regression/recovery | `src/lls.cpp`, `src/lls/irreversible.cpp`, `src/lls/reversible.cpp` |
| Shared bounded nonlinear optimizer | `src/solver.cpp` |
| TAC batches, CLI, Python binding | `src/tac_batch.cpp`, `src/tac_fit_main.cpp`, `src/python_bindings.cpp` |
| Python model descriptions, fitting, evaluation, results | `kinepet/models.py`, `kinepet/fitting.py`, `kinepet/evaluation.py`, `kinepet/results.py` |

See [the architecture guide](docs/architecture.md) for dispatch paths, coefficient definitions, and the boundary between model equations and fitting algorithms.

## Input format

The solver works on flattened TAC batches.

Required files:

- `tacs.npy`: shape `(T, N)`
- `time.npy`: shape `(T,)`, `(T, 1)`, or `(T, N)`, in minutes
- `aif.npy`: shape `(T,)`, `(T, 1)`, or `(T, N)`

Interpretation of `aif.npy`:

- `(T,)` or `(T, 1)`: one shared AIF for all TACs
- `(T, N)`: one AIF per TAC

Interpretation of `time.npy`:

- `(T,)` or `(T, 1)`: one shared time vector for all TACs
- `(T, N)`: one time vector per TAC
- when `aif.npy` is `(T, N)`, `time.npy` must also be `(T, N)` so `time[:, i]`, `aif[:, i]`, and `tacs[:, i]` describe the same sample

Optional:

- `weights.npy`: shape `(T,)`, `(T, 1)`, or `(T, N)`

Time is always interpreted directly as minutes. The solver does not infer seconds from large values and does not perform automatic unit conversion. Convert frame times to minutes before fitting.

## Build and install

Source installs require:

- Python
- `uv`
- CMake
- a C++20 compiler toolchain

The Python API uses a compiled nanobind extension, so CMake is required when installing from source. It is not only needed for the standalone CLI. If you install a prebuilt wheel, CMake should not be required.

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

## Python API

Model objects make the kinetic equations and blood-fraction setting explicit:

```python
import kinepet

model = kinepet.Reversible2TCM(vb=kinepet.FittedVB())
fit = kinepet.fit_tacs(tacs, time, aif, model=model)

# A known nonzero vascular contribution:
model = kinepet.Irreversible2TCM(vb=kinepet.FixedVB(0.05))
fit = kinepet.fit_one_tac(tac, time, aif, model=model)

# Use FixedVB(0.0) to omit the vascular contribution.
```

Both model types default to `FittedVB()`. `FixedVB` accepts finite fractions in
`[0, 1]`. The same objects work with `evaluate_model()` and `default_bounds()`.
For a fixed model, evaluation uses its known blood fraction; supplying a different
`vB` raises `ValueError`. For a fitted model, evaluation uses the supplied numeric
`vB`, defaulting to `0.05`.

Existing string-based calls remain supported, with their existing defaults:

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

With a model object, set blood fraction on the object and omit `fit_vb` and
`fixed_vb`; combining both forms raises `ValueError`. For legacy string models,
omitting those options retains `fit_vb=True` and `fixed_vb=0.05`. Internally,
`None` marks an omitted option so conflicting configurations can be detected.

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

The source of truth for bounds is `src/parameters.cpp`.

Current defaults:

- `K1`: `[0, 10]`
- `k2`: `[0, 10]`
- `k3`: `[0, 5]`
- `k4`: `[0, 1]` when active
- `vB`: `[0, 1]` when fitted
- `delay`: `[-0.2, 0.2]` minutes when fitted
- `dispersion`: `[0, 0.1]` minutes when fitted

## Scripts

Generate TACs from a simulated parameter CSV and export fitted parameters:

```bash
uv run --group extra python scripts/fit_from_csv.py input.csv fitted.csv --model rev --fit-vb
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

Build C++ targets as above, then also run:

```bash
ctest --test-dir build --output-on-failure
```

The tests cover model objects and legacy calls, fitted/fixed/zero blood fraction,
shared/per-TAC timing and AIFs, optional AIF parameters, repeated-root forward
kinetics, and direct reversible coefficient recovery. C++ warm starts are checked
against pre-refactor baselines. After building the CLI, pytest also checks CLI/API
parity across 36 model/input configurations. If the CLI is elsewhere, set
`KINEPET_CLI` to its executable path; otherwise those tests skip when it is absent.

## Notes

- The nonlinear solver is custom C++, not SciPy TRF copied line-for-line.
- There is exactly ONE forward-model implementation in this repo: the compiled C++
  entrypoint `src/tcm_model.cpp` dispatches to `src/models/irreversible.cpp` or
  `src/models/reversible.cpp`, reachable from Python as `kinepet.evaluate_model(...)`.
  It integrates the piecewise-linear AIF with the exact exponential step and
  agrees with an ODE integration (`scipy.integrate.solve_ivp`) to machine
  precision, including for very small rate constants. The previous
  `scripts/kinetics.py` NumPy model has been removed.
- Synthetic data is generated with that same C++ model, so a noiseless batch
  fits back to its parameters (median relative error ~1e-4). Generating with a
  separate implementation would make the dataset a different target than the
  one the solver minimizes and would bias every recovered parameter.
- Delay and dispersion support are available, but simultaneous estimation is
  still more stable than the core kinetic parameters on difficult curves.
