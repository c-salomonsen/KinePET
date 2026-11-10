# Python API

The public Python API is exposed from `kinepet`.

```python
import kinepet
```

Construct a model instance, then pass it to a fitting or evaluation function:

```python
model = kinepet.Reversible2TCM(vb=kinepet.FittedVB())
fit = kinepet.fit_tacs(tacs, time, aif, model=model)
```

Here `tacs`, `time`, and `aif` are measured input arrays; a complete runnable
example is available in [Getting Started](../getting-started.md).

Primary functions:

| Function | Purpose |
| --- | --- |
| `fit_tacs()` | Fit a batch of TACs arranged as `(T, N)` |
| `fit_one_tac()` | Fit one TAC arranged as `(T,)` |
| `evaluate_model()` | Evaluate a model curve for supplied parameters |
| `default_bounds()` | Return active default parameter bounds |
| `default_frame_weights()` | Return residual weights derived from sample spacing |
| `status_code_to_name()` | Convert numeric solver status to a string |

Primary result types:

| Type | Returned by |
| --- | --- |
| `BatchFitResult` | `fit_tacs()` |
| `SingleTacFitResult` | `fit_one_tac()` |

Python handles array normalization, shape validation, model descriptions, and result containers. Numerical input validation and fitting run in the compiled extension.

## Model descriptions

| Class | Purpose |
| --- | --- |
| `Reversible2TCM(vb=...)` | Reversible kinetics with K1, k2, k3, and k4. |
| `Irreversible2TCM(vb=...)` | Irreversible kinetics with K1, k2, and k3. |
| `FittedVB()` | Estimate blood fraction during fitting. |
| `FixedVB(value)` | Hold blood fraction at a known value; zero omits vascular activity. |

Both kinetic classes default to `FittedVB()`. Model objects hold configuration;
results are returned by `fit_tacs` or `fit_one_tac`. The same model instance can
be used to inspect bounds and evaluate curves. See
[Models and Parameters](../models.md) for configuration rules and string-based
API compatibility.
