# Python API

The public Python API is exposed from `kinepet`.

```python
import kinepet
```

Primary functions:

| Function | Purpose |
| --- | --- |
| `fit_tacs()` | Fit a batch of TACs arranged as `(T, N)` |
| `fit_one_tac()` | Fit one TAC arranged as `(T,)` |
| `evaluate_model()` | Evaluate a model curve for supplied parameters |
| `default_bounds()` | Return active default parameter bounds |
| `default_frame_weights()` | Return frame-duration-style default weights |
| `status_code_to_name()` | Convert numeric solver status to a string |

Primary result types:

| Type | Returned by |
| --- | --- |
| `BatchFitResult` | `fit_tacs()` |
| `SingleTacFitResult` | `fit_one_tac()` |

Python handles array normalization, shape validation, model descriptions, and result containers. Numerical input validation and fitting run in the compiled extension.

## Model descriptions

```python
model = kinepet.Reversible2TCM(vb=kinepet.FixedVB(0.05))
fit = kinepet.fit_tacs(tacs, time, aif, model=model)
```

`Reversible2TCM` and `Irreversible2TCM` both accept `FittedVB()` (the default) or
`FixedVB(value)`. See [Models and Parameters](../models.md) for configuration,
evaluation behavior, and compatibility with existing string-model calls.
