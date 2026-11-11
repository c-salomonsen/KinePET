# Development

## Local setup

```bash
uv sync --group dev
```

This installs the package in editable mode and builds the compiled Python extension. Source builds require CMake and a C++20 compiler toolchain.

## Run tests

```bash
uv run pytest -q
```

## Build C++ targets

Use this when you also want the standalone `tcm_fit_tacs` CLI target:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)')
cmake --build build
ctest --test-dir build --output-on-failure
```

## Build documentation

Install documentation dependencies:

```bash
uv sync --group docs
```

Build the static site:

```bash
uv run mkdocs build
```

Serve locally:

```bash
uv run mkdocs serve
```

The site is configured for `https://kinepet.salodev.no/`. Relevant
pushes to `main` and pull requests validate the documentation without publishing.
Successful package releases deploy the documentation built from the release
commit. Manual publication requires an explicit source tag, branch, or commit. See
[the hosting setup](releasing.md#documentation-hosting) for deployment instructions.

## Validation checklist

After solver, binding, or CLI changes, run:

```bash
uv run pytest -q
cmake -S . -B build -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)')
cmake --build build
ctest --test-dir build --output-on-failure
```

If CLI-facing behavior changes, also run `tcm_fit_tacs` on a small synthetic TAC batch.

## Reading and extending the code

Start with [the architecture guide](architecture.md). It maps public calls to
model dispatch, LLS regression/recovery, shared nonlinear optimization, and
forward-model evaluation.

The standalone CMake build enables CTest by default. Wheel builds disable C++
test targets through `tool.scikit-build.cmake.define` in `pyproject.toml`.
Python tests run independently; CLI parity tests additionally require a built
CLI in `build/` or an executable path supplied through `KINEPET_CLI`.

Before a release, build both distributions and check their contents:

```bash
uv build
```

The package version is defined in `pyproject.toml`. Follow
[the release procedure](releasing.md) to prepare a versioned distribution.
After editing C++ sources, rebuild the editable extension with
`uv sync --group dev --reinstall-package kinepet` before running Python tests.
