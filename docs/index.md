# KinePET

KinePET fits dynamic PET time–activity curves (TACs) with a bounded nonlinear
solver and linear least-squares initialization. The Python API and the standalone
`tcm_fit_tacs` CLI share the same compiled C++ fitting core.

Choose a kinetic model and a blood-fraction setting:

| Python model | Tissue parameters | CLI model name |
| --- | --- | --- |
| `Irreversible2TCM` | K1, k2, k3 | `irr` |
| `Reversible2TCM` | K1, k2, k3, k4 | `rev` |

Both models fit the blood fraction by default. Use `FixedVB(value)` for a known
fraction or `FixedVB(0.0)` to omit the vascular contribution.

For measured arrays `tacs`, `time`, and `aif`:

```python
import kinepet

model = kinepet.Reversible2TCM(vb=kinepet.FittedVB())
fit = kinepet.fit_tacs(tacs, time, aif, model=model, threads=8)

print(fit.K1.shape)
print(fit.status)
```

The TAC layout is `(T, N)`, where `T` is the number of time samples and `N` is the
number of curves. Time is expressed in minutes. The model object specifies the
equations and blood-fraction setting; `fit_tacs` performs the fit and returns
parameters and diagnostics for each curve.

Start with [Getting Started](getting-started.md) for a complete synthetic example.
[Models and Parameters](models.md) explains model selection and scientific
conventions. The [Python API](python-api/index.md) covers fitting, forward
evaluation, and result access. Existing `model="irr"` and `model="rev"` calls
remain supported.
