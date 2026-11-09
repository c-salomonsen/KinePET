# Architecture and Reading Guide

KinePET has two kinetic model types and one shared nonlinear optimizer. A model
describes tissue kinetics and whether the vascular blood fraction is fitted or
fixed. LLS initialization and nonlinear optimization are separate operations.

## Model descriptions

Python:

```python
model = kinepet.Reversible2TCM(vb=kinepet.FittedVB())
model = kinepet.Irreversible2TCM(vb=kinepet.FixedVB(0.05))
```

C++:

```cpp
const tcm::Model model = tcm::Reversible2TCM{tcm::FixedVB{0.05}};
const tcm::AifOptions aif_options{};
const auto bounds = tcm::default_bounds(model, aif_options);
const auto initial = tcm::warm_start(model, aif_options, bounds, time, aif, tac);
```

`src/model.hpp` defines `Irreversible2TCM`, `Reversible2TCM`, `FittedVB`, and
`FixedVB`. `Model` is a `std::variant` holding either kinetic type: dispatch uses
its actual type, without virtual base classes or a solver inheritance hierarchy.
Both kinetic types carry a `BloodFraction` setting in their `vb` member.

A fixed blood fraction can be nonzero. `FixedVB{0.0}` is the special case without
a vascular contribution. Delay and dispersion are independent optional fit
parameters, grouped in `AifOptions` as named `delay` and `dispersion` settings.

Python model objects are immutable. The wrappers translate them to the existing
nanobind arguments; the binding translates those arguments into typed C++ models
once, before batch fitting. The CLI similarly translates its flags after parsing,
so argument order cannot change the model's blood-fraction setting.

## From a public call to a fitted TAC

| Step | Source | What to read |
| --- | --- | --- |
| Public Python API | `kinepet/__init__.py` | Exports only; fitting, evaluation, models, and results have their own modules |
| Array normalization | `kinepet/_validation.py` | Shared/per-TAC shape checks; no kinetic calculations |
| Python fitting | `kinepet/fitting.py` | `fit_tacs`, `fit_one_tac`, and model-option resolution |
| Native interface | `src/python_bindings.cpp` | `make_solver_options`, array conversion, result ownership |
| CLI interface | `src/tac_fit_main.cpp` | Flag parsing, model configuration, NPY input/output |
| Batch fitting | `src/tac_batch.cpp` | Column selection, default weights, OpenMP loop, per-TAC result storage |
| Single-TAC fitting | `src/solver.cpp` | `fit_tac` → `fit_tac_impl` |
| Initialization | `src/solver.cpp` | `choose_initial_parameters`: explicit guess, valid LLS warm start, or default guess; then bounds clipping |
| Nonlinear optimization | `src/solver.cpp` | Initial residuals, finite-difference Jacobian, LM steps, acceptance, convergence status |
| Forward evaluation | `src/tcm_model.cpp` | AIF handling → tissue-model dispatch → vascular mixing |

The optimizer uses the same residual/Jacobian/step logic for both kinetic models.
It does not reconstruct kinetic parameters from LLS coefficients at each
iteration. That recovery belongs to initialization alone.

## Finding the LLS equations

Start at `warm_start` in `src/lls.cpp`. It checks dimensions, prepares the shared
AIF integrals, initializes optional parameters, and dispatches to
`lls::estimate_linear_parameters` for the actual kinetic type. The model-specific
function selects the fitted or fixed blood-fraction formulation in the same file.

| Formulation | Regression function | Recovery function | Source |
| --- | --- | --- | --- |
| Irreversible, fitted blood fraction | `warm_start_irreversible_fitted_vb` | `recover_irreversible_fitted_vb` | `src/lls/irreversible.cpp` |
| Irreversible, fixed blood fraction | `warm_start_irreversible_fixed_vb` | `recover_irreversible_fixed_vb` | `src/lls/irreversible.cpp` |
| Reversible, fitted blood fraction | `warm_start_reversible_fitted_vb` | `recover_reversible_fitted_vb` | `src/lls/reversible.cpp` |
| Reversible, fixed blood fraction | `warm_start_reversible_fixed_vb` | `recover_reversible_fixed_vb` | `src/lls/reversible.cpp` |

The small shared implementation in `src/lls/regression.cpp` performs cumulative
trapezoidal integration and ridge regression. The formulation files contain the
regressor ordering and coefficient definitions beside the recovery equations.

For the reversible fitted-blood-fraction case, write `I` for integration and
`m` for measured activity. The design columns and coefficients are:

```text
X = [I(m), I(I(m)), C_A, I(C_A), I(I(C_A))]
m = a1*X1 + a2*X2 + a3*X3 + a4*X4 + a5*X5

alpha = k2+k3+k4
beta  = k2*k4
gamma = K1*(k3+k4)

a1 = -alpha
a2 = -beta
a3 = vB
a4 = vB*alpha + (1-vB)*K1
a5 = vB*beta  + (1-vB)*gamma

d0 = a4+a1*a3
s  = (a5+a2*a3)/d0

vB = a3
K1 = d0/(1-a3)
k2 = -a1-s
k4 = -a2/k2
k3 = s-k4
```

The fixed-blood-fraction formulations first remove the known vascular
contribution, `q=(m-vB*C_A)/(1-vB)`, then regress tissue activity against four
columns for reversible kinetics or three for irreversible kinetics. Their
coefficient ordering differs from the fitted-blood-fraction formulations and is
specified explicitly in each recovery function.

Preserved numerical details matter when comparing initialization:

- Integrals start at the first supplied time sample; no missing initial interval
  is inserted by LLS.
- Regression uses ridge `1e-6` on the normal-equation diagonal.
- Recovery retains the existing `1e-8` thresholds and denominator epsilons.
- The fixed-blood-fraction LLS preprocessing retains its `[0, 0.95]` clamp.
  The final parameters still use the configured fixed fraction through parameter
  unpacking. This existing distinction was preserved, rather than changed during
  the readability refactor.
- LLS uses the input AIF directly; it does not solve delay or dispersion. Those
  parameters retain their initial/fixed values for nonlinear refinement.
- Invalid kinetic recovery rejects the warm start. Accepted parameters are
  clipped to the configured bounds.

## Finding the forward equations

`src/tcm_model.cpp` is the single public compiled forward-evaluation path. It
preserves the existing AIF thresholding and transformations, calls the appropriate
`kinetics::evaluate_tissue` overload, and applies:

```text
m = (1-vB)*q + vB*C_A
```

| Responsibility | Source |
| --- | --- |
| Irreversible tissue response | `src/models/irreversible.cpp` |
| Reversible biexponential response and repeated-root RK4 fallback | `src/models/reversible.cpp` |
| Exact exponential convolution over a piecewise-linear AIF segment | `src/models/convolution.cpp` |
| AIF sampling and delay/dispersion transformation | `src/aif.cpp` |
| Active parameter ordering, packing, bounds, defaults, and Ki | `src/parameters.cpp` |
| Default residual weights | `src/frame_weights.cpp` |

Both tissue overloads return `q=f+b`, expressed per extravascular tissue volume.
They share the convolution helper. Vascular mixing is applied once after tissue
evaluation, regardless of whether `vB` was fitted or fixed.

The forward evaluator uses the numeric parameters it is given. The solver's
packing/unpacking enforces fixed parameters during optimization. The Python
wrapper additionally resolves a model object's `FixedVB` during standalone
forward evaluation and rejects a conflicting explicit `vB`.

Synthetic generation and reference plotting must continue to call
`kinepet.evaluate_model`; do not introduce a second NumPy kinetics implementation.

## Verification and continuing the refactor

`tests/cpp/test_core.cpp` checks pre-refactor warm-start baselines for both models,
fitted/fixed/zero blood fraction, and zero/positive first sample times. It also
checks exact reversible coefficient recovery, invalid warm-start rejection, and
parameter ordering/packing across all optional-parameter combinations.

`tests/test_models.py` checks typed/legacy API parity and repeated-root behavior.
`tests/test_cli.py` checks CLI/API parity with shared/per-TAC inputs and optional
AIF parameters. Existing binding and small-rate regression tests remain in place.

Keep future changes focused on one responsibility. Update this map when moving
code; preserve the shared optimizer and compiled forward path unless a functional
change explicitly requires otherwise. The refactor leaves the package version
at `0.5.0.dev0`; unreleased changes are recorded in `CHANGELOG.md`.
