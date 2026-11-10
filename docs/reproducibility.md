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

## Synthetic benchmark

```bash
uv run python scripts/benchmark.py --num-tacs 128 --repeats 3 --threads 1 --output output/benchmark.json
```

The command generates noiseless data using the compiled forward model, fits
both models over matched K1/k2/k3/vB ranges, and saves timings, native-grid RMSE,
parameter errors, failure counts, solver settings, seed, versions, and machine
information. It performs an untimed warm-up. Record compiler/version, build
flags, OpenMP availability, and hardware with the JSON for published timings.
A requested thread count does not establish that OpenMP was enabled in the build.

This benchmark measures consistency and local throughput on generated TACs.
It includes only KinePET; comparisons with other packages require separate
measurements on matched inputs. Physical validation is covered by the ODE tests.
`scripts/wheel_smoke.py` provides a smaller installed-package check:
both models and blood-fraction settings must converge with RMSE below `1e-5`
and recover the selected parameters within 1% (absolute tolerance `1e-4`).

## VOI comparison export

The exporter defaults to KinePET only, so it works without private packages:

```bash
uv run python scripts/extract_voi_reference_fits.py \
  --raw-npz-glob 'path/to/raw/*.npz' --voi-glob 'path/to/VOI_*.pkl' \
  --npz-time-unit minutes --voi-time-unit seconds --methods cpp
```

Specify the actual units of each source. No magnitude-based conversion is used.
NPZ object arrays and pickle inputs must come from a trusted source: loading
pickles can execute code.

To include PyPET, add `pypet` to `--methods` and provide a reviewed local module
with `--pypet-module`. Alternatively provide `--pypet-sha256` for the download;
the default URL pins commit `8d552908f5f2accb2d026f6c3081dbf5bb961096`.
Install that reference's dependencies separately. Metadata records the actual
module SHA-256, path, and download URL. Inspect reference code before executing it.

The `lls` comparison uses a separate reference package, `lls` version 0.5.0,
distributed through a private package index. A public archive for this reference
is not recorded in this repository. Access to the reference distribution is
required to reproduce the LLS comparison. Install it separately before adding
`lls` to `--methods`. KinePET, its tests, documentation, and synthetic benchmark
do not depend on this package.

For the `cpp` method, CSV `status` contains the numeric status code.
`error_value` retains each method's original metric and `error_kind`:
KinePET RMSE on native samples, PyPET root-sum-of-squares on interpolated samples,
and LLS MSE on interpolated samples. These values must not be ranked directly.
`common_rmse_native_grid` evaluates every method's recovered parameters
with `kinepet.evaluate_model` against the same native TAC. Invalid/out-of-domain
parameters leave that metric NaN. This comparison measures parameter quality
under KinePET's common forward model; it is not the reference method's own loss.

## Paper analysis provenance

The paper analysis used neither delay nor dispersion fitting. The candidate
software version is `v0.4.0`, commit
`2309d229890d8f99bf91287f9b71bfcb6e7dfdc0`; its correspondence to the installed
analysis artifact has not yet been verified. No versioned software archive DOI
is recorded here.

The AIF aliasing fix in `0.4.1` affects optional delay/dispersion transformations.
Fits with both corrections fixed to zero do not use that transformation path.
The invalid-input checks in `0.4.1` apply to all configurations.

A reproducible analysis record includes the exact software artifact or revision,
solver options, weights, time and activity units, preprocessing, reference
versions and hashes, seeds, compiler and OpenMP configuration, hardware, and
input access conditions. The [release procedure](releasing.md) describes archive
and citation metadata requirements.
