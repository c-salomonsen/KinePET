# Batch Fitting

Use `kinepet.fit_tacs()` for the main TAC-batch workflow.

```python
fit = kinepet.fit_tacs(
    tacs,
    time,
    aif,
    weights=None,
    model="rev",
    fit_vb=True,
    fixed_vb=0.05,
    fit_delay=False,
    fixed_delay=0.0,
    fit_dispersion=False,
    fixed_dispersion=0.0,
    threads=0,
    max_iterations=40,
    gradient_tolerance=1e-6,
    step_tolerance=1e-6,
    cost_tolerance=1e-9,
    fd_step=1e-4,
    lambda_initial=1e-2,
    use_linear_warm_start=True,
)
```

## Inputs

| Argument | Shape or type | Description |
| --- | --- | --- |
| `tacs` | `(T, N)` | TAC matrix. Each column is one TAC. |
| `time` | `(T,)`, `(T, 1)`, or `(T, N)` | Frame times in minutes. KinePET does not infer seconds or rescale large values, so callers must convert seconds to minutes before fitting. |
| `aif` | `(T,)`, `(T, 1)`, or `(T, N)` | Arterial input function. A vector or single-column array is shared by all TACs. `(T, N)` provides one AIF per TAC and requires `time` to also be `(T, N)`. |
| `weights` | `None`, `(T,)`, `(T, 1)`, or `(T, N)` | Optional residual weights. Shared and per-TAC semantics match `aif`. |
| `model` | Model object, `"irr"`, or `"rev"` | Kinetic model and blood-fraction configuration. |
| `threads` | `int` | OpenMP thread count when available. `0` lets the backend choose. |

## Shared and per-TAC timing

For a shared AIF, pass a shared time vector:

```python
fit = kinepet.fit_tacs(tacs, time, aif, model="rev")
```

For per-TAC AIFs, pass per-TAC times:

```python
fit = kinepet.fit_tacs(
    tacs,
    time_matrix,  # shape (T, N)
    aif_matrix,   # shape (T, N)
    model="rev",
)
```

In the per-TAC case, column `i` is fitted as `time_matrix[:, i]`, `aif_matrix[:, i]`, and `tacs[:, i]`.

All time columns are interpreted in minutes, even when a scan extends beyond 60 minutes.

## Optional fitted parameters

By default KinePET fits `vB`, does not fit delay, and does not fit dispersion.

```python
model = kinepet.Reversible2TCM(vb=kinepet.FixedVB(0.05))
fit = kinepet.fit_tacs(tacs, time, aif, model=model)
```

With a model object, omit `fit_vb` and `fixed_vb`; specifying either raises
`ValueError`. The defaults below describe legacy string calls. Their Python
signature uses `None` to represent an omitted blood-fraction option.

| Option | Default | Meaning |
| --- | --- | --- |
| `fit_vb` | `True` | Fit blood volume fraction. |
| `fixed_vb` | `0.05` | Used when `fit_vb=False`. |
| `fit_delay` | `False` | Fit AIF delay in minutes. |
| `fixed_delay` | `0.0` | Used when `fit_delay=False`. |
| `fit_dispersion` | `False` | Fit AIF dispersion in minutes. |
| `fixed_dispersion` | `0.0` | Used when `fit_dispersion=False`. Must be non-negative. |

Delay and dispersion are available, but simultaneous estimation can be less stable than fitting the core kinetic parameters on difficult curves.

## Return value

`fit_tacs()` returns `BatchFitResult`. Parameter arrays are flat `(N,)` arrays:

```python
fit = kinepet.fit_tacs(tacs, time, aif, model="rev")

print(fit.K1)
print(fit.k2)
print(fit.k3)
print(fit.k4)
print(fit.vB)
print(fit.Ki)
print(fit.rmse)
print(fit.status)
```

For `model="irr"`, `fit.k4` is `None`. `fit.delay` and `fit.dispersion` are also `None` unless the corresponding parameter is fitted or fixed to a nonzero value.

Use `as_dict()` when you want to iterate over returned fields:

```python
arrays = fit.as_dict()

for name, value in arrays.items():
    print(name, value)
```
