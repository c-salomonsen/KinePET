# Data Format

KinePET fits flattened TAC batches.

## Required arrays

| Array | Shape | Meaning |
| --- | --- | --- |
| `tacs` / `tacs.npy` | `(T, N)` | TAC matrix. Each column is one TAC. |
| `time` / `time.npy` | `(T,)`, `(T, 1)`, or `(T, N)` | Time vector or per-TAC time matrix, in minutes. |
| `aif` / `aif.npy` | `(T,)`, `(T, 1)`, or `(T, N)` | Arterial input function. |

## Optional arrays

| Array | Shape | Meaning |
| --- | --- | --- |
| `weights` / `weights.npy` | `(T,)`, `(T, 1)`, or `(T, N)` | Residual weights. |

## Shared and per-TAC semantics

For `time`, `aif`, and `weights`:

- `(T,)` means one shared vector for every TAC.
- `(T, 1)` also means one shared vector for every TAC.
- `(T, N)` means one column per TAC.

When `aif` is `(T, N)`, `time` must also be `(T, N)`. Column `i` is fitted as `time[:, i]`, `aif[:, i]`, and `tacs[:, i]`.

## Time units

Time values are always interpreted directly as minutes. KinePET does not infer seconds from large values and does not perform automatic unit conversion. Convert frame times to minutes before fitting.

## Outputs

Python returns dataclass wrappers. The CLI writes `.npy` files. In both cases outputs are flat per-TAC values with shape `(N,)`.

Main outputs:

| Output | Availability |
| --- | --- |
| `K1` | Always |
| `k2` | Always |
| `k3` | Always |
| `k4` | `rev` only |
| `vB` | Always |
| `delay` | When fitted or fixed to a nonzero value |
| `dispersion` | When fitted or fixed to a nonzero value |
| `Ki` | Always |
| `rmse` | Always |
| `weighted_cost` | Python API |
| `iterations` | Always |
| `status` | Always |

Image preprocessing and reconstruction of parameter maps from per-TAC outputs are handled by the calling application.
