# CLI

The standalone CLI is `tcm_fit_tacs`. It fits `.npy` TAC-batch directories.

## Build from source

Install development dependencies:

```bash
uv sync --group dev
```

`uv sync --group dev` already builds the Python extension through CMake. To build the standalone CLI target as well, run CMake explicitly:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)')
cmake --build build
```

Source builds require CMake and a C++20 compiler. These are needed for the Python extension as well as the CLI.

## Run

```bash
./build/tcm_fit_tacs --input-dir INPUT --output-dir OUTPUT --model rev --threads 8
```

Input directory:

```text
INPUT/
  tacs.npy
  time.npy
  aif.npy
```

`time.npy` may be `(T,)`, `(T, 1)`, or `(T, N)` and is always interpreted in minutes. KinePET does not infer seconds or rescale large values. If `aif.npy` is `(T, N)`, `time.npy` must also be `(T, N)` so each TAC column is fitted with its matching AIF and sample times.

Optional weights can be passed with `--weights-file`:

```bash
./build/tcm_fit_tacs \
  --input-dir INPUT \
  --output-dir OUTPUT \
  --model irr \
  --weights-file INPUT/weights.npy
```

## Options

| Option | Description |
| --- | --- |
| `--input-dir DIR` | Directory containing `tacs.npy`, `time.npy`, and `aif.npy`. |
| `--output-dir DIR` | Directory for result files. |
| `--model rev\|irr` | Model to fit. |
| `--fit-vb 0\|1` | Fit blood volume fraction. Default: `1`. |
| `--fixed-vb VALUE` | Fixed blood volume when `--fit-vb 0`. Default: `0.05`. |
| `--fit-delay 0\|1` | Fit AIF delay. Default: `0`. |
| `--fixed-delay VALUE` | Fixed AIF delay in minutes when delay is not fitted. Default: `0.0`. |
| `--fit-dispersion 0\|1` | Fit AIF dispersion. Default: `0`. |
| `--fixed-dispersion VALUE` | Fixed AIF dispersion in minutes when dispersion is not fitted. Default: `0.0`. |
| `--threads N` | OpenMP threads when available. |
| `--max-iter N` | Maximum trust-region iterations. Default: `40`. |
| `--weights-file PATH` | Optional `.npy` weights array. |
| `--no-linear-warm-start` | Disable linear least-squares initialization. |

## Output files

The CLI writes flat `(N,)` arrays:

| File | Availability |
| --- | --- |
| `K1.npy` | Always |
| `k2.npy` | Always |
| `k3.npy` | Always |
| `k4.npy` | `rev` only |
| `vB.npy` | Always |
| `delay.npy` | When delay is fitted or fixed to a nonzero value |
| `dispersion.npy` | When dispersion is fitted or fixed to a nonzero value |
| `Ki.npy` | Always |
| `rmse.npy` | Always |
| `iterations.npy` | Always |
| `status.npy` | Always, numeric status codes |
| `run.txt` | Manifest for the run |

`status.npy` contains numeric status codes. See [Results and Status](python-api/results.md) for the status table.
