# Validation and benchmarks

Run from the source checkout after installing `python -m pip install -e ".[test]"`.
The scripts use synthetic or analytic fields and write results under
`validation/output/` by default. For repeat runs use a fresh output directory
when the script preserves existing results.

| Script | Study |
| --- | --- |
| `run_validation.py` | Integration and interpolation error, open faces, and tangent contacts |
| `benchmark_tracing.py` | RK2 and RK45 at matched trajectory error targets |
| `benchmark_olshevsky.py` | General and near-null seeds in the analytic 14-null field, with optional DOP853 references |
| `benchmark_threading_layers.py` | Output equality and timings across Numba backends |
| `diagnostic_sensitivity.py` | Effects of saved-point spacing on sampling, diagnostics, and winding |

```sh
NUMBA_NUM_THREADS=2 python validation/benchmark_olshevsky.py \
  --n 64 --threads 1 2 --orders C F --reference both \
  --out validation/output/olshevsky
```

See [the fixture guide](olshevsky/README.md) for the field definition,
provenance, and memory requirements. For a simulation snapshot, record the reader
version, centering, grid axes and periods, seed set, field precision, integration
controls, and endpoint codes before comparing trajectories.
