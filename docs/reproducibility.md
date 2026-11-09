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
  RK4 fallback uses relative tolerance `3e-5`. This is a tested range, not a
  claim of machine precision for every possible input.

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

This benchmark measures consistency and local throughput. It does not reproduce
the paper's data, establish clinical accuracy, or compare speed against another
package. `scripts/wheel_smoke.py` provides a smaller installed-package check:
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

The `lls` comparison requires the maintainer's separate `lls` package (previously
version 0.5.0 from a private package index). It is not required for KinePET,
tests, documentation, or the synthetic benchmark. Full reproduction of that
comparison still requires a publicly archived, licensed reference distribution;
no substitute implementation is silently installed. Add `lls` to `--methods`
only after installing the intended reference version.

Existing CSV columns are preserved. C++ `status` contains the numeric status
code. `error_value` retains each method's original metric and `error_kind`:
KinePET RMSE on native samples, PyPET root-sum-of-squares on interpolated samples,
and LLS MSE on interpolated samples. These values must not be ranked directly.
The added `common_rmse_native_grid` evaluates every method's recovered parameters
with `kinepet.evaluate_model` against the same native TAC. Invalid/out-of-domain
parameters leave that metric NaN. This comparison measures parameter quality
under KinePET's common forward model; it is not the reference method's own loss.

## Paper provenance

The maintainer reports that the paper used the published version, believed to
be `v0.4.0` (commit `2309d229890d8f99bf91287f9b71bfcb6e7dfdc0`), with neither delay
nor dispersion fitting. Confirm the installed artifact/version against the
analysis environment before archiving the final paper record. The AIF aliasing
fix affects optional delay/dispersion transformations, not fits with both fixed
to zero. Invalid-input handling should still be checked against the paper inputs.

The new development version is separate from the paper version. Keep the
original results and tag intact. The paper archive should record the exact
revision/artifact, options, weights, time and activity units, preprocessing,
reference versions/hashes, seeds, compiler/OpenMP/hardware, and permitted input
access. Add the software archive DOI and paper DOI to `CITATION.cff` when available.
