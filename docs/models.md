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
