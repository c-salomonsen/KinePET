# Single TAC Fitting

Use `kinepet.fit_one_tac()` when you have one curve.

```python
fit = kinepet.fit_one_tac(
    tac,
    time,
    aif,
    model="irr",
    fit_vb=True,
)
```

`tac`, `time`, `aif`, and optional `weights` must all be one-dimensional arrays with the same length.

The options match `fit_tacs()`, except there is no `threads` argument because only one TAC is fitted.

## Accessing results

`fit_one_tac()` returns `SingleTacFitResult`. Fields are scalars instead of arrays:

```python
print(fit.K1)
print(fit.k2)
print(fit.k3)
print(fit.vB)
print(fit.Ki)
print(fit.rmse)
print(fit.status)
print(fit.status_code)
```

For `model="rev"`, `fit.k4` contains the reversible exchange parameter. For `model="irr"`, `fit.k4` is `None`.

## Round trip with `evaluate_model`

```python
tac = kinepet.evaluate_model(
    time,
    aif,
    model="rev",
    K1=0.12,
    k2=0.16,
    k3=0.09,
    k4=0.04,
    vB=0.05,
)

fit = kinepet.fit_one_tac(tac, time, aif, model="rev", max_iterations=200)
```
