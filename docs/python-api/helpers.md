# Helpers

## `evaluate_model`

`evaluate_model` computes a curve without fitting. The examples below assume
one-dimensional `time` and `aif` arrays:

```python
import kinepet

model = kinepet.Reversible2TCM(vb=kinepet.FittedVB())
pred = kinepet.evaluate_model(
    time,
    aif,
    model=model,
    K1=0.12,
    k2=0.16,
    k3=0.09,
    k4=0.04,
    vB=0.05,
    delay=0.0,
    dispersion=0.0,
)
```

`time` and `aif` must be one-dimensional arrays with the same length. `time`, `delay`, and `dispersion` are interpreted in minutes. The returned array has shape `(T,)`.

For `FittedVB()`, `vB` supplies the numeric blood fraction for this evaluation;
omitting it uses `0.05`. No parameters are estimated by `evaluate_model`.

For `FixedVB(value)`, omit `vB` to use the fraction stored in the model:

```python
fixed_model = kinepet.Reversible2TCM(vb=kinepet.FixedVB(0.05))
pred = kinepet.evaluate_model(time, aif, model=fixed_model,
                              K1=0.12, k2=0.16, k3=0.09, k4=0.04)
```

An explicit `vB` that differs from `FixedVB.value` raises `ValueError`. The model
constructor uses lowercase `vb` for the setting object; the evaluation function
uses `vB` for the numeric value. For `Irreversible2TCM`, omit `k4`.

## `default_bounds`

Inspect the active parameter bounds used by the solver:

```python
bounds = kinepet.default_bounds(
    model,
    fit_delay=True,
    fit_dispersion=True,
)

print(bounds["K1"])
print(bounds["k4"])
print(bounds["vB"])
```

The returned dictionary contains only parameters estimated for this
configuration. `FixedVB` removes `vB` from that dictionary; `Irreversible2TCM`
omits `k4`. Bounds for delay and dispersion appear only when their fit flags
are enabled.

```python
fixed_bounds = kinepet.default_bounds(fixed_model)
print(fixed_bounds.keys())  # K1, k2, k3, k4
```

With a model object, omit `fit_vb` and `fixed_vb`. This helper returns the default
bounds for inspection; it does not change the model or configure a fit.

Current defaults:

| Parameter | Bounds |
| --- | --- |
| `K1` | `[0, 10]` |
| `k2` | `[0, 10]` |
| `k3` | `[0, 5]` |
| `k4` | `[0, 1]` when active |
| `vB` | `[0, 1]` when fitted |
| `delay` | `[-0.2, 0.2]` minutes when fitted |
| `dispersion` | `[0, 0.1]` minutes when fitted |

## `default_frame_weights`

Generate the solver's default frame weights for a time vector in minutes:

```python
weights = kinepet.default_frame_weights(time)
fit = kinepet.fit_tacs(tacs, time, aif, weights=weights, model=model)
```

If `weights=None`, the batch fitter uses default frame weights internally. Time values are not rescaled before weights are computed.
