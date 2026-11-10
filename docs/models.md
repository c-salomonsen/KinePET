# Models and Parameters

A model object selects the tissue equations and determines whether the vascular
blood fraction is fitted or fixed.

| Python class | Tissue parameters fitted | Equivalent string name |
| --- | --- | --- |
| `Irreversible2TCM` | K1, k2, k3; k4 is zero | `"irr"` |
| `Reversible2TCM` | K1, k2, k3, k4 | `"rev"` |

## Explicit model objects

```python
import kinepet

reversible = kinepet.Reversible2TCM(vb=kinepet.FittedVB())
irreversible = kinepet.Irreversible2TCM(vb=kinepet.FixedVB(0.05))
no_vascular_contribution = kinepet.Reversible2TCM(vb=kinepet.FixedVB(0.0))
```

Both kinetic classes default to `FittedVB()`: `Reversible2TCM()` estimates K1, k2,
k3, k4, and vB when passed to a fitting function. Model objects are immutable
configurations and can be reused across fits. Construct another object to change
the model or blood fraction. Fitted values are returned in a result object.

Pass a model instance to `fit_tacs`, `fit_one_tac`, `evaluate_model`, or
`default_bounds`. The fitting functions perform LLS initialization and nonlinear
optimization using the shared solver; the model objects have no fitting methods.

### Blood-fraction settings

| Setting | Behavior during fitting |
| --- | --- |
| `FittedVB()` | Estimate vB separately for each TAC. |
| `FixedVB(0.05)` | Hold vB at 0.05 for every TAC in the call. |
| `FixedVB(0.0)` | Set vB to zero, omitting the vascular contribution. |

`FixedVB(value)` requires a finite value in `[0, 1]`. Its default value is `0.05`;
writing the value explicitly makes the assumption visible. The constructor
argument is lowercase `vb`, and it accepts a `FittedVB` or `FixedVB` object:

```python
model = kinepet.Reversible2TCM(vb=kinepet.FixedVB(0.05))
```

With a model object, omit `fit_vb` and `fixed_vb` from fitting and bounds calls.
Supplying either legacy option as well raises `ValueError`, even when it agrees
with the model's setting.

### Forward evaluation

`evaluate_model` computes a curve from supplied parameters. For a model with
`FittedVB()`, supply the numeric blood fraction through the case-sensitive `vB`
argument; omission uses `0.05`. `FittedVB()` does not estimate anything during
forward evaluation.

A model with `FixedVB(value)` supplies its own fraction, so `vB` can be omitted.
An explicit `vB` must equal the model's value; a conflicting value raises
`ValueError`. See [forward evaluation examples](python-api/helpers.md#evaluate_model).

### Compatibility with string-based calls

The following configurations are equivalent for fitting:

| Model object | String-based arguments |
| --- | --- |
| `Reversible2TCM()` | `model="rev", fit_vb=True` |
| `Irreversible2TCM()` | `model="irr", fit_vb=True` |
| `Reversible2TCM(vb=FixedVB(0.05))` | `model="rev", fit_vb=False, fixed_vb=0.05` |
| `Irreversible2TCM(vb=FixedVB(0.0))` | `model="irr", fit_vb=False, fixed_vb=0.0` |

String-based calls retain effective defaults `model="rev"`, `fit_vb=True`, and
`fixed_vb=0.05` when these arguments are omitted. Model objects make the same
choices explicit without combining a model name and blood-fraction flags.

Delay and dispersion remain independent fitting options on `fit_tacs` and
`fit_one_tac`. They are disabled by default for both model types. Their numeric
values are passed separately to `evaluate_model` when generating a curve.

Canonical parameter names:

| Parameter | Description |
| --- | --- |
| `K1` | Delivery rate parameter |
| `k2` | Efflux rate parameter |
| `k3` | Binding or trapping rate parameter |
| `k4` | Reversible exchange parameter, active for `rev` |
| `vB` | Blood volume fraction |
| `delay` | Optional AIF delay in minutes |
| `dispersion` | Optional AIF dispersion in minutes |
| `Ki` | Net influx macroparameter |

For the irreversible model, `k4` is not active and Python result objects expose `k4=None`.

## Bounds

The source of truth for bounds is `src/parameters.cpp`. Use `kinepet.default_bounds()` to inspect active bounds from Python:

```python
kinepet.default_bounds(kinepet.Reversible2TCM(), fit_delay=True, fit_dispersion=True)
```

Default active bounds:

| Parameter | Bounds |
| --- | --- |
| `K1` | `[0, 10]` |
| `k2` | `[0, 10]` |
| `k3` | `[0, 5]` |
| `k4` | `[0, 1]` |
| `vB` | `[0, 1]` |
| `delay` | `[-0.2, 0.2]` minutes |
| `dispersion` | `[0, 0.1]` minutes |

## Parameter order note

KinePET's exported names are canonical. Some external comparison methods use different parameter orders, so normalize names before comparing methods.

## Scientific conventions

Time, delay, and dispersion are in minutes; k2, k3, and k4 are in min⁻¹.
K1 has tissue-concentration / input-concentration / minute units (conventionally
mL blood or plasma per mL or g tissue per minute, depending on the inputs).
`vB` is dimensionless. Supply compatible TAC and AIF activity units.

Predictions are **point samples at the supplied times**, not frame-averaged
integrals. The AIF is piecewise linear, with a linear ramp from zero at time zero
when the first sample time is positive. The tissue compartments initially contain
zero tracer. AIF values below `1e-5` in the supplied activity units are set to zero
before optional transformation; this includes negative values. This absolute
threshold means rescaling activity can affect very small AIF samples.

The measured signal is `(1-vB) * (C1+C2) + vB * AIF`. The same supplied/transformed
AIF drives tissue delivery and vascular mixing. The library does not independently
model plasma and whole-blood curves, metabolites, or hematocrit. Apply and report
any required corrections consistently before fitting; this convention may not
represent every tracer's physiology.

The optional transformation is
`max(0, AIF(t + delay) + dispersion * dAIF/dt(t + delay))`.
A positive delay samples later measured AIF values; a negative delay samples
earlier ones. Sampling before time zero gives zero, and sampling beyond the
last measurement holds the endpoint constant with zero derivative. At interior
knots the right-hand slope is used; the last endpoint has zero derivative.
Dispersion here is a first-order derivative correction, not convolution with an
exponential dispersion kernel. Simultaneous estimation of delay/dispersion and
kinetic parameters can be poorly identified; both options are off by default.

The minimized cost is `0.5 * sum((weights * (prediction - TAC))**2)`.
Weights multiply residuals, so they are squared in the objective. Defaults are
the square roots of sample spacings (centered spacings internally, adjacent
spacings at endpoints, widths floored at `1e-6` minutes). They are not a Poisson
variance model or measured frame durations. RMSE is unweighted on native samples.

`Ki = K1*k3/(k2+k3)` is returned for both models (zero when the denominator is
at most `1e-12`). It is the net irreversible influx macroparameter when `k4=0`.
For reversible kinetics it is that same algebraic combination, not a claim of
permanent trapping or long-time net influx. Interpret it alongside k4 and the
tracer model.

Convergence does not establish parameter identifiability or physiological
validity. Inspect bounds, residual structure, acquisition duration, and sensitivity
before interpreting fitted parameters. The repeated-eigenvalue reversible branch
uses numerical RK4 integration; see [validation scope](reproducibility.md).
