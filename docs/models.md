# Models and Parameters

KinePET supports two public model names:

| Name | Model |
| --- | --- |
| `irr` | Irreversible two-tissue compartment model |
| `rev` | Reversible two-tissue compartment model |

## Explicit model objects

```python
from kinepet import Reversible2TCM, Irreversible2TCM, FittedVB, FixedVB

reversible = Reversible2TCM(vb=FittedVB())
irreversible = Irreversible2TCM(vb=FixedVB(0.05))
no_vascular_contribution = Reversible2TCM(vb=FixedVB(0.0))
```

Both kinetic types default to `FittedVB()`. They are immutable descriptions of
the equations and blood-fraction estimation; optimization remains in the shared
compiled solver. `FixedVB(value)` requires a finite value in `[0, 1]`.

Pass these objects to `fit_tacs`, `fit_one_tac`, `evaluate_model`, or
`default_bounds`. With a model object, omit legacy `fit_vb` / `fixed_vb` options;
combining both configuration forms raises `ValueError`.

For evaluation, a `FixedVB` object supplies the blood fraction. A conflicting
numeric `vB` raises `ValueError`. For a `FittedVB` object, provide the numeric
fraction through `vB` (default `0.05`), since evaluation does not estimate it.

Legacy `model="irr"` / `model="rev"` calls retain their prior behavior and defaults.
The optional AIF delay and dispersion settings are independent of either model.

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
kinepet.default_bounds(model="rev", fit_vb=True, fit_delay=True, fit_dispersion=True)
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
