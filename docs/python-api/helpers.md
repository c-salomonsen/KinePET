# Helpers

## `evaluate_model`

Evaluate a model TAC from explicit parameters:

```python
pred = kinepet.evaluate_model(
    time,
    aif,
    model="rev",
    K1=0.12,
    k2=0.16,
    k3=0.09,
    k4=0.04,
    vB=0.05,
    delay=0.0,
    dispersion=0.0,
)
```

`time` and `aif` must be one-dimensional arrays with the same length. The returned array has shape `(T,)`.

## `default_bounds`

Inspect the active parameter bounds used by the solver:

```python
bounds = kinepet.default_bounds(
    model="rev",
    fit_vb=True,
    fit_delay=True,
    fit_dispersion=True,
)

print(bounds["K1"])
print(bounds["k4"])
print(bounds["vB"])
```

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

Generate the solver's default frame weights for a time vector:

```python
weights = kinepet.default_frame_weights(time)
fit = kinepet.fit_tacs(tacs, time, aif, weights=weights, model="rev")
```

If `weights=None`, the batch fitter uses default frame weights internally.
