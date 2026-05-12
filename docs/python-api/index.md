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

The API is intentionally light. Input validation and dataclass result wrappers live in Python; the nonlinear fitting core runs in the compiled extension.
