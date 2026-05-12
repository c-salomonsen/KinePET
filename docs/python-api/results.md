# Results and Status

## `BatchFitResult`

`fit_tacs()` returns:

| Field | Type | Description |
| --- | --- | --- |
| `K1` | `np.ndarray` | Delivery rate parameter, shape `(N,)`. |
| `k2` | `np.ndarray` | Efflux rate parameter, shape `(N,)`. |
| `k3` | `np.ndarray` | Binding or trapping rate parameter, shape `(N,)`. |
| `k4` | `np.ndarray \| None` | Reversible model parameter. `None` for `irr`. |
| `vB` | `np.ndarray` | Blood volume fraction, shape `(N,)`. |
| `delay` | `np.ndarray \| None` | AIF delay when active. |
| `dispersion` | `np.ndarray \| None` | AIF dispersion when active. |
| `Ki` | `np.ndarray` | Net influx macroparameter, shape `(N,)`. |
| `rmse` | `np.ndarray` | Root mean square residual, shape `(N,)`. |
| `weighted_cost` | `np.ndarray` | Final weighted objective value, shape `(N,)`. |
| `iterations` | `np.ndarray` | Solver iteration count, shape `(N,)`. |
| `status` | `np.ndarray` | Human-readable status strings, shape `(N,)`. |
| `status_code` | `np.ndarray` | Numeric status codes, shape `(N,)`. |
| `fitted_tacs` | `int` | Number of TACs with non-invalid fit results. |

`len(fit)` returns `N`.

## `SingleTacFitResult`

`fit_one_tac()` returns the same conceptual fields as scalars. Optional fields such as `k4`, `delay`, and `dispersion` are `None` when inactive.

## Status values

| Code | Python name | Meaning |
| --- | --- | --- |
| `0` | `failed` | Fit failed. |
| `1` | `gradient_converged` | Gradient convergence. |
| `2` | `step_converged` | Proposed step became tiny. This is a weaker success signal than gradient or cost convergence. |
| `3` | `cost_converged` | Cost convergence. |
| `4` | `max_iterations` | Maximum iteration count reached. |
| `5` | `invalid_input` | Input was invalid for that TAC. |

Convert a code manually with:

```python
name = kinepet.status_code_to_name(2)
```

For batch results, `fit.status` already contains strings and `fit.status_code` preserves the original numeric codes.
