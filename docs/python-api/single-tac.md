# Single TAC Fitting

`kinepet.fit_one_tac()` fits one measured curve. The examples below use the
one-dimensional arrays `tac`, `time`, and `aif`.

```python
import kinepet

model = kinepet.Irreversible2TCM(vb=kinepet.FittedVB())
fit = kinepet.fit_one_tac(
    tac,
    time,
    aif,
    model=model,
)
```

`tac`, `time`, `aif`, and optional `weights` must all be one-dimensional arrays with the same length. `time` is always interpreted directly as minutes.

The options match `fit_tacs()`, except there is no `threads` argument because only one TAC is fitted.

For a known blood fraction, construct a fixed-fraction model:

```python
fixed_model = kinepet.Irreversible2TCM(vb=kinepet.FixedVB(0.05))
fit = kinepet.fit_one_tac(tac, time, aif, model=fixed_model)
```

Model objects own their blood-fraction setting; omit the legacy `fit_vb` and
`fixed_vb` options when using one.

Convert seconds to minutes before calling `fit_one_tac()`.

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

For `Reversible2TCM`, `fit.k4` contains the reversible exchange parameter. For `Irreversible2TCM`, `fit.k4` is `None`. Fixed blood fractions are still returned in `fit.vB`.

## Round trip with `evaluate_model`

```python
reversible = kinepet.Reversible2TCM(vb=kinepet.FittedVB())
tac = kinepet.evaluate_model(
    time,
    aif,
    model=reversible,
    K1=0.12,
    k2=0.16,
    k3=0.09,
    k4=0.04,
    vB=0.05,
)

fit = kinepet.fit_one_tac(tac, time, aif, model=reversible, max_iterations=200)
```

The forward call supplies the numeric `vB`; the fitting call estimates it because
`reversible` uses `FittedVB()`. The model configuration is unchanged by either call.
