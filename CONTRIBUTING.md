# Contributing to KinePET

Report bugs and discuss changes through the
[issue tracker](https://github.com/c-salomonsen/KinePET/issues).
For fitting problems, include the version or commit, model, options, time units,
status codes, operating system, and a small synthetic reproducer. Do not post
patient data or credentials.

Set up development with `uv sync --group dev`. Read
[the architecture guide](docs/architecture.md) before changing the solver.
Use the compiled `kinepet.evaluate_model` for synthetic data and application
kinetics. Independent reference equations belong in validation tests only.

Before proposing numerical or API changes, run:

```bash
uv sync --group dev --reinstall-package kinepet
cmake -S . -B build -DPython_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"
cmake --build build
ctest --test-dir build --output-on-failure
uv run pytest -q
```

Explain the problem, the resulting behavior, and the validation in your pull
request. Numerical fixes should include a regression test that exposes the
original defect. Update the relevant API documentation and changelog.

The latest public release is the primary support target; fixes for older
versions are considered case by case. Support is provided through issues on a
best-effort basis, without guaranteed response times. For a suspected security
issue, use GitHub's private vulnerability reporting when enabled; otherwise
contact the maintainer through their GitHub profile before posting details.
