# Batch Fitting

`kinepet.fit_tacs()` fits each column of a TAC matrix using the selected model.
The examples below assume measured arrays `tacs`, `time`, and `aif`; see
[Getting Started](../getting-started.md) for a complete synthetic example.

```python
import kinepet

model = kinepet.Reversible2TCM(vb=kinepet.FittedVB())
fit = kinepet.fit_tacs(
    tacs,
    time,
    aif,
    weights=None,
    model=model,
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
fit = kinepet.fit_tacs(tacs, time, aif, model=model)
```

For per-TAC AIFs, pass per-TAC times:

```python
fit = kinepet.fit_tacs(
    tacs,
    time_matrix,  # shape (T, N)
    aif_matrix,   # shape (T, N)
    model=model,
)
```

In the per-TAC case, column `i` is fitted as `time_matrix[:, i]`, `aif_matrix[:, i]`, and `tacs[:, i]`.

All time columns are interpreted in minutes, even when a scan extends beyond 60 minutes.

## Optional fitted parameters

Blood fraction is configured on the model. Both kinetic model classes default
to `FittedVB()`. To use a known value for every TAC in the batch:

```python
fixed_model = kinepet.Reversible2TCM(vb=kinepet.FixedVB(0.05))
fit = kinepet.fit_tacs(tacs, time, aif, model=fixed_model)
```

Use `FixedVB(0.0)` to omit the vascular contribution. With a model object, omit
`fit_vb` and `fixed_vb`; specifying either also raises `ValueError`. See
[Models and Parameters](../models.md#compatibility-with-string-based-calls) for
legacy string-based equivalents.

Delay and dispersion are configured on the fitting call for either model:

| Option | Default | Meaning |
| --- | --- | --- |
| `fit_delay` | `False` | Estimate AIF delay in minutes. |
| `fixed_delay` | `0.0` | Delay used when `fit_delay=False`. |
| `fit_dispersion` | `False` | Estimate AIF dispersion in minutes. |
| `fixed_dispersion` | `0.0` | Dispersion used when `fit_dispersion=False`; must be finite and nonnegative. |

For example, to estimate delay while holding dispersion at zero:

```python
fit = kinepet.fit_tacs(tacs, time, aif, model=model, fit_delay=True)
```

Estimating delay or dispersion together with tissue rates can reduce parameter
identifiability. See [scientific conventions](../models.md#scientific-conventions)
for the transformation definition and time units.

## Return value

`fit_tacs()` returns `BatchFitResult`. Parameter arrays are flat `(N,)` arrays:

```python
fit = kinepet.fit_tacs(tacs, time, aif, model=model)

print(fit.K1)
print(fit.k2)
print(fit.k3)
print(fit.k4)
print(fit.vB)
print(fit.Ki)
print(fit.rmse)
print(fit.status)
```

For `Irreversible2TCM`, `fit.k4` is `None`. `fit.delay` and `fit.dispersion` are also `None` unless the corresponding parameter is fitted or fixed to a nonzero value.

Use `as_dict()` when you want to iterate over returned fields:

```python
arrays = fit.as_dict()

for name, value in arrays.items():
    print(name, value)
```
