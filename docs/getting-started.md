# Getting Started

## Install for development

Clone the repository, then install the editable package and development dependencies:

```bash
uv sync --group dev
```

This installs KinePET from source and builds the compiled nanobind extension used by the Python API.

Source installs require:

- Python supported by the project
- `uv`
- CMake
- a C++20 compiler toolchain

Source installation builds the `_core` extension used by the Python API. Prebuilt wheels include this extension and require no local CMake installation or compiler.

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

time = np.array([0.0, 0.0833, 0.1667, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
aif = 12.0 * np.exp(-time / 0.35) + 1.5 * np.exp(-time / 8.0)

tacs = np.stack(
    [
        kinepet.evaluate_model(time, aif, model="irr", K1=0.10, k2=0.12, k3=0.06, vB=0.04),
        kinepet.evaluate_model(time, aif, model="irr", K1=0.14, k2=0.18, k3=0.08, vB=0.06),
    ],
    axis=1,
)

fit = kinepet.fit_tacs(tacs, time, aif, model="irr", threads=2)

print(fit.K1)
print(fit.k2)
print(fit.k3)
print(fit.vB)
print(fit.status)
```

`fit` is a `BatchFitResult`. Each fitted parameter is a one-dimensional NumPy array with shape `(N,)`.

## Build the CLI

The CLI is built from the same CMake project:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)')
cmake --build build
```

Then run:

```bash
./build/tcm_fit_tacs --input-dir INPUT --output-dir OUTPUT --model rev --threads 8
```

See [CLI](cli.md) for accepted inputs, options, and output files.
