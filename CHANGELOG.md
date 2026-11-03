# Changelog

## Unreleased

### Added

- Explicit Python `Irreversible2TCM` and `Reversible2TCM` descriptions with
  `FittedVB()` or `FixedVB(value)`. Accepted by fitting, forward evaluation, and
  bounds helpers alongside the existing `irr` / `rev` names.
- Direct C++ tests for reversible coefficient recovery, parameter layout,
  rejected warm starts, and pre-refactor initialization baselines.
- Python model-configuration and CLI/API parity tests, including shared/per-TAC
  inputs and optional AIF parameters. C++ numerical tests run in CLI CI.
- Architecture and equation-location guide in `docs/architecture.md`.

### Changed

- C++ models carry their blood-fraction setting explicitly; optional AIF settings
  are grouped rather than passed as repeated boolean/value arguments.
- Model-specific LLS regressions and kinetic parameter recovery have separate,
  named implementations with coefficient definitions beside the equations.
- Tissue kinetics, convolution, AIF transformations, vascular mixing, parameter
  layout, bounds, and default frame weights have distinct source files.
- Internal per-curve C++ entrypoints are named `fit_tac` and
  `fit_tac_with_initial_guess`, replacing the legacy voxel terminology.
- Python public exports, model descriptions, fitting, evaluation, validation, and
  result containers have separate modules.
- Distribution builds omit C++ test targets; standalone CMake builds retain CTest.

Existing string-model calls, effective defaults, result shapes, CLI flags and
filenames, status semantics, bounds, and numerical algorithms are preserved.
With a model object, configure blood fraction on that object; combining it with
legacy `fit_vb` / `fixed_vb` options raises `ValueError`. A standalone evaluation
also rejects a numeric `vB` that conflicts with the model's `FixedVB`.

The version remains `0.4.0` pending a release decision.
