# KinePET

[![Release](https://img.shields.io/github/v/release/c-salomonsen/KinePET?logo=github)](https://github.com/c-salomonsen/KinePET/releases)
[![Changelog](https://img.shields.io/badge/changelog-view-blue?logo=keepachangelog&logoColor=white)](https://github.com/c-salomonsen/KinePET/blob/main/changelog.md)
[![Tests](https://github.com/c-salomonsen/KinePET/actions/workflows/tests.yml/badge.svg)](https://github.com/c-salomonsen/KinePET/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white)](https://kinepet.salodev.no/)
[![License](https://img.shields.io/github/license/c-salomonsen/KinePET)](https://github.com/c-salomonsen/KinePET?tab=AGPL-3.0-1-ov-file)
[![DOI](https://zenodo.org/badge/1231964802.svg)](https://doi.org/10.5281/zenodo.24132815)

KinePET fits dynamic PET time–activity curves (TACs) with reversible and
irreversible two-tissue compartment models. A C++ bounded nonlinear solver uses
a linear least-squares warm start and supports fitting TAC batches through
Python or the standalone `tcm_fit_tacs` CLI.

Blood fraction can be fitted, fixed to a known value, or omitted by fixing it to
zero. Delay and dispersion correction are optional. Image preprocessing and
reconstruction of parameter maps belong in the calling application.

## Installation

Python 3.11 or later is required. Install from PyPI in a virtual environment
with pip:

```bash
python -m pip install kinepet
```

Or with uv:

```bash
uv venv --python 3.11
uv pip install kinepet
```

For an existing uv project, use `uv add kinepet`. See
[Getting Started](docs/getting-started.md#install) for environment setup and
activation instructions.

Release CI builds and tests wheels for CPython 3.11–3.13 on Linux, Windows, and
macOS. To install from a source checkout, run from the repository root:

```bash
python -m pip install .
```

Source builds require a C++20 compiler and CMake (the Python build backend can
install CMake). Prebuilt wheels do not require a compiler. Runtime dependencies
are limited to NumPy. Development dependencies resolve from public PyPI:

```bash
uv sync --group dev
```

See [the development guide](docs/development.md) for build and test instructions.

## Working example

```python
import numpy as np
import kinepet

time = np.array([0., .0833, .1667, .25, .3333, .5, .75, 1., 1.5,
                 2., 3., 5., 8., 12., 20., 30., 45.])  # minutes
aif = 12 * np.exp(-time / .3) + 1.5 * np.exp(-time / 6)
model = kinepet.Reversible2TCM(vb=kinepet.FittedVB())

tac = kinepet.evaluate_model(
    time, aif, model=model, K1=.09, k2=.22, k3=.11, k4=.06, vB=.04,
)
fit = kinepet.fit_one_tac(tac, time, aif, model=model, max_iterations=200)
print(fit.K1, fit.k2, fit.k3, fit.k4, fit.vB)
print(fit.status, fit.rmse)

# Batch input has shape (number of time samples, number of TACs).
batch = kinepet.fit_tacs(np.column_stack([tac, tac]), time, aif, model=model)
```

Use `Irreversible2TCM` for irreversible kinetics, `FixedVB(0.05)` for a known
blood fraction, or `FixedVB(0.0)` for no vascular contribution. Both kinetic
models default to `FittedVB()`. Legacy calls such as
`fit_tacs(tacs, time, aif, model="rev", fit_vb=True)` remain supported. With a
model object, specify blood fraction on the object and omit `fit_vb`/`fixed_vb`.

## Inputs and interpretation

- `tacs`: `(T, N)`; each column is one TAC.
- `time`, `aif`, and optional `weights`: `(T,)`, `(T, 1)`, or `(T, N)`.
- A vector or single column is shared across TACs. Per-TAC AIFs require per-TAC
  times, and shared AIFs require shared times.
- Time, delay, and dispersion are in **minutes**. There is no automatic unit
  conversion. Times must be finite, nonnegative, and strictly increasing.
- Fits require at least four samples and finite TAC/AIF values. Weights are
  finite nonnegative residual multipliers; each TAC needs at least one positive
  weight. Zero-weight samples must still contain finite inputs.

Invalid data yields `invalid_input` (`status_code=5`) for the affected TAC;
other columns continue fitting. Invalid global solver options raise an error.
Numerical evaluation failure yields `failed` (`0`), not convergence. Inspect
status, residuals, and parameter plausibility: a small optimization step alone
(`2`) is a weaker success signal than gradient (`1`) or cost (`3`) convergence.
`fitted_tacs` counts attempted fits with valid input, including failed and
iteration-limited fits; it is not a success count.

Outputs are flat per-TAC parameters (`K1`, `k2`, `k3`, optional `k4`, `vB`,
optional `delay`/`dispersion`, `Ki`), RMSE, weighted cost, iterations, and status.
Python exposes both status names and numeric codes. See
[scientific conventions](docs/models.md) for sampling, vascular mixing,
AIF treatment, weights, and the interpretation of `Ki`.

## CLI

```bash
cmake -S . -B build -DPython_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"
cmake --build build
./build/tcm_fit_tacs --input-dir INPUT --output-dir OUTPUT --model rev --threads 4
```

The input directory contains `tacs.npy`, `time.npy`, and `aif.npy` with the same
shapes and units as the Python API. Supply weights through `--weights-file`.
The CLI writes per-TAC `.npy` outputs and `run.txt`; weighted cost is currently
Python-only. See [CLI options and filenames](docs/cli.md).

## Validation

```bash
uv run pytest -q
ctest --test-dir build --output-on-failure
```

The tests cover LLS recovery, independent ODE checks, invalid inputs, typed and
legacy API calls, and CLI/API parity. Rebuild the extension after C++ edits with
`uv sync --group dev --reinstall-package kinepet`. CLI tests require the built
executable, or `KINEPET_CLI` pointing to it.

Paper-specific comparisons, analysis notebooks, and the previous simulation
scripts are maintained separately in `kinepet-paper-validation`. The library
retains numerical regression tests and installed-package checks. See
[validation and reproducibility](docs/reproducibility.md) for their scope and
for the information needed to identify the software used in an analysis.

## Code organization

The fitting path is Python/CLI → batch orchestration → per-TAC solver → model evaluation.

| Responsibility | Location |
| --- | --- |
| Model types and blood fraction | `src/model.hpp` |
| Tissue kinetics and exact convolution | `src/models/` |
| AIF transformation and vascular mixing | `src/aif.cpp`, `src/tcm_model.cpp` |
| Parameter layout and bounds | `src/parameters.cpp` |
| LLS dispatch, regression, parameter recovery | `src/lls.cpp`, `src/lls/` |
| Nonlinear optimization and input checks | `src/solver.cpp`, `src/validation.hpp` |
| TAC batches, CLI, Python binding | `src/tac_batch.cpp`, `src/tac_fit_main.cpp`, `src/python_bindings.cpp` |
| Python API | `kinepet/` |

[The architecture guide](docs/architecture.md) maps model dispatch and recovery
coefficients. Applications and synthetic-data scripts use the single compiled
forward implementation through `kinepet.evaluate_model`.

## License

Copyright © 2026 Christian Salomonsen.

KinePET is licensed under the GNU Affero General Public License,
version 3 only (AGPL-3.0-only). See [LICENSE](LICENSE) for the full terms.
