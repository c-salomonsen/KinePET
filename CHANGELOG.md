# Changelog

## 0.4.2 (2026-11-11)

This release updates packaging, release automation, and documentation hosting.
The fitting algorithms and public model API are unchanged from 0.4.1.

### Changed

- Build distributions once, upload them to TestPyPI for review, then publish the
  same files to official PyPI after approval through the `pypi` environment.
  Saved artifact IDs and SHA-256 checksums preserve the files between uploads.
- Generate draft GitHub release notes from the matching changelog entry and
  publish the release after the PyPI upload succeeds.
- Deploy the validated release documentation to GitHub Pages after successful
  PyPI and GitHub release publication, retaining `https://kinepet.salodev.no/`.
  Development pushes validate documentation without updating the live site;
  manual publication requires an explicit source tag, branch, or commit.
- Run one Linux/Python configuration for routine pushes and pull requests.
  Reserve the full supported Python test matrix and platform distribution builds
  for release tags and manual runs.
- Document Trusted Publishing setup, release review and recovery, and GitHub
  Pages configuration. Update citation metadata and add the Zenodo DOI badge.

### Fixed

- Replace the retired `macos-14` runner with `macos-15` and explicitly build
  Apple Silicon (`arm64`) wheels. The unavailable runner previously prevented
  the distribution workflow from reaching publication.

## 0.4.1 (2026-11-11)

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

- Move paper-specific comparison scripts and tests, standalone simulations,
  and the synthetic benchmark to the companion validation repository. Library
  numerical tests, CLI parity checks, and wheel validation remain in KinePET.
- Group contributor and maintenance documentation separately from user guides.

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

The model API refactor preserves string-model calls, effective defaults, result
shapes, CLI flags and filenames, status semantics, bounds, and numerical algorithms.
With a model object, configure blood fraction on that object; combining it with
legacy `fit_vb` / `fixed_vb` options raises `ValueError`. A standalone evaluation
also rejects a numeric `vB` that conflicts with the model's `FixedVB`.

### Fixes and distribution updates

- Reject nonfinite inputs, invalid time grids/weights, and invalid solver options
  in the shared C++ path. Preserve per-TAC invalid status 5 in batch outputs;
  guard objective, derivative, and linear-step overflow against false convergence.
- Fix in-place AIF transformation for negative delays and dispersion.
- Add numerical input regressions and independent compartment ODE checks.
- Resolve public development dependencies without a private index. Prepare
  Trusted Publishing with release metadata/license gates and numerical wheel tests.
- Add dependency notices, contribution guidance, citation metadata, scientific
  conventions, validation scope, and release instructions.
- Add AGPL-3.0-only licensing and SPDX notices to project source and build files.
