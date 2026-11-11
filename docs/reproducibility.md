# Validation and reproducibility

## Numerical checks

The public test suite includes:

- Exact reversible LLS coefficient recovery, independent of quadrature and optimization.
- Pre-refactor warm-start baselines and Python/CLI parity.
- Invalid-input and numerical-overflow handling, including mixed valid/invalid batches.
- Pure vascular delay/dispersion checks against interpolation of unchanged AIF samples.
- Independent integration of the compartment mass-balance ODEs in
  `tests/test_physical_reference.py`. Integration restarts at each AIF knot.
  The test covers both kinetic models, positive/zero first sample times, small
  rates, larger rates, and a repeated-root case. The general comparison uses
  relative tolerance `2e-9` and absolute tolerance `2e-10`; the repeated-root
  RK4 fallback uses relative tolerance `3e-5`. These tolerances apply to the
  parameter combinations and sampling protocol defined in the test.

## Installed-package checks

`scripts/wheel_smoke.py` checks both kinetic models with fitted and fixed blood
fractions. The selected synthetic cases must converge with RMSE below `1e-5`
and recover their parameters within 1% (absolute tolerance `1e-4`). These checks
run against the built wheels in CI.

The CLI tests create temporary TAC inputs and compare CLI outputs against the
Python API across model, blood-fraction, and AIF configurations. Synthetic
curves used in tests and documentation are small verification examples.

## Paper analysis

Study-specific comparisons, datasets, notebooks, and the previous standalone
simulation and benchmark scripts are maintained in the separate
`kinepet-paper-validation` repository.
The KinePET package and its test suite do not depend on that repository or on
the external PyPET and LLS reference implementations.

A reproducible analysis record identifies the exact software artifact or
revision, solver options, weights, time and activity units, preprocessing,
reference versions and hashes, seeds, compiler and OpenMP configuration,
hardware, and input access conditions. See the [release procedure](releasing.md)
for software archive and citation metadata requirements.
