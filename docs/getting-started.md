# Getting Started

## Install

KinePET is available from PyPI and requires Python 3.11 or later. Choose either
`uv` or `pip` to install it in a virtual environment.

### With uv

```bash
uv venv --python 3.11
uv pip install kinepet
```

For an existing uv project, use `uv add kinepet` to add it to the project's
dependencies instead.

### With pip

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the package:

```bash
python -m pip install kinepet
```

The uv-created environment can be activated with the same commands. Run the
examples below with that environment's Python interpreter.

Prebuilt wheels include the compiled extension and require no local compiler.
If no compatible wheel is available, installation builds from source and
requires a C++20 compiler; the build backend installs CMake as needed.

### From a source checkout

Clone the repository, then install the editable package and development dependencies:

```bash
uv sync --group dev
```

This installs KinePET from source and builds the compiled nanobind extension used by the Python API.

Source installs require:

- Python 3.11 or later
- `uv`
- CMake
- a C++20 compiler toolchain

See [Development](development.md) for build and test instructions.

## First batch fit

`fit_tacs()` expects:

- `tacs`: shape `(T, N)`
- `time`: shape `(T,)`, `(T, 1)`, or `(T, N)`, in minutes
- `aif`: shape `(T,)`, `(T, 1)`, or `(T, N)`

When `aif` is `(T, N)`, `time` must also be `(T, N)` so each TAC is fitted with its matching AIF and sample-time column.

KinePET always interprets `time` directly as minutes. Convert seconds to minutes before fitting.

```python
import numpy as np
import kinepet

time = np.array([0., .0833, .1667, .25, .3333, .5, .75, 1., 1.5,
                 2., 3., 5., 8., 12., 20., 30., 45.])
aif = 12 * np.exp(-time / .3) + 1.5 * np.exp(-time / 6)
model = kinepet.Reversible2TCM(vb=kinepet.FittedVB())

# Generate two synthetic TACs with known parameters.
tacs = np.column_stack([
    kinepet.evaluate_model(time, aif, model=model,
                          K1=.09, k2=.22, k3=.11, k4=.06, vB=.04),
    kinepet.evaluate_model(time, aif, model=model,
                          K1=.14, k2=.18, k3=.08, k4=.04, vB=.06),
])

# Fit the tissue parameters and blood fraction of each TAC.
fit = kinepet.fit_tacs(tacs, time, aif, model=model,
                       threads=2, max_iterations=200)

print(fit.K1)
print(fit.k2)
print(fit.k3)
print(fit.k4)
print(fit.vB)
print(fit.status)
print(fit.rmse)
```

`fit` is a `BatchFitResult`. Each parameter and diagnostic is a one-dimensional
array with one entry per TAC. Inspect the [status and residuals](python-api/results.md)
before interpreting the fitted parameters.

`FittedVB()` tells the fitting function to estimate blood fraction. Forward
evaluation requires a numeric fraction, supplied as `vB` in the two simulation
calls above. The model object holds the configuration; it does not store the
simulated values or fitted results.

For a known blood fraction, construct a model with that value:

```python
fixed_model = kinepet.Reversible2TCM(vb=kinepet.FixedVB(0.04))
fixed_fit = kinepet.fit_one_tac(tacs[:, 0], time, aif, model=fixed_model,
                               max_iterations=200)
print(fixed_fit.vB)  # 0.04
```

Use `Irreversible2TCM` for irreversible kinetics, where k4 is fixed to zero.
Use `FixedVB(0.0)` with either model to omit the vascular contribution. See
[Models and Parameters](models.md) for the complete set of configurations and
compatibility with string-based calls.

## Build the CLI

The CLI is built from a source checkout. Complete the development installation
above, then run these commands from the repository root:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)')
cmake --build build
```

Then run:

```bash
./build/tcm_fit_tacs --input-dir INPUT --output-dir OUTPUT --model rev --threads 8
```

See [CLI](cli.md) for accepted inputs, options, and output files.
