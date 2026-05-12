# Models and Parameters

KinePET supports two public model names:

| Name | Model |
| --- | --- |
| `irr` | Irreversible two-tissue compartment model |
| `rev` | Reversible two-tissue compartment model |

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

The source of truth for bounds is `src/tcm_model.cpp`. Use `kinepet.default_bounds()` to inspect active bounds from Python:

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
