# KinePET

KinePET is a TAC-based bounded nonlinear fitter for dynamic PET two-tissue compartment models.

The main interface is the Python package `kinepet`. A standalone C++ CLI, `tcm_fit_tacs`, is also available for fitting `.npy` TAC batches from the command line.

Supported user-facing models:

| Model | Meaning |
| --- | --- |
| `irr` | Irreversible two-tissue compartment model |
| `rev` | Reversible two-tissue compartment model |

KinePET works on flattened time-activity curve batches. The expected TAC layout is `(T, N)`, where `T` is the number of time points and `N` is the number of curves.

```python
import kinepet

fit = kinepet.fit_tacs(
    tacs,
    time,
    aif,
    model="rev",
    fit_vb=True,
    threads=8,
)

print(fit.K1.shape)
print(fit.status)
```

Start with [Getting Started](getting-started.md) for installation and a complete first fit, or go directly to the [Python API](python-api/index.md).
