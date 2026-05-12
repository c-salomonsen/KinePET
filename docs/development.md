# Development

## Local setup

```bash
uv sync --group dev
```

This installs the package in editable mode and builds the compiled Python extension. Source installs require CMake and a C++20 compiler toolchain; they are not only CLI prerequisites.

## Run tests

```bash
uv run pytest -q
```

## Build C++ targets

Use this when you also want the standalone `tcm_fit_tacs` CLI target:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)')
cmake --build build
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

The site is configured for `https://kinepet.salodev.no/`.

## Validation checklist

After solver, binding, or CLI changes, run:

```bash
uv run pytest -q
cmake -S . -B build -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)')
cmake --build build
```

If CLI-facing behavior changes, also run `tcm_fit_tacs` on a small synthetic TAC batch.
