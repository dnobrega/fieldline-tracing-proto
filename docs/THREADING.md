# Numba threading

Numba selects a threading layer when its parallel runtime initializes.
Available layers depend on the Numba build and system libraries. To request
OpenMP (`omp`), TBB (`tbb`), or `workqueue`, set `NUMBA_THREADING_LAYER`
before starting Python. Restart notebook kernels after changing this setting.
TBB and OpenMP require their corresponding runtime when the build supports
them.

```sh
NUMBA_THREADING_LAYER=tbb NUMBA_NUM_THREADS=8 python validation/benchmark_olshevsky.py \
  --n 64 --threads 1 2 4 8 --reference none --out validation/output/tbb
```

`NUMBA_NUM_THREADS` sets the maximum pool size before import.
`numba.set_num_threads(n)` sets the active thread count within that limit.
Call `numba.threading_layer()` after the first parallel solve to check which
backend initialized.

The comparison script runs separate processes with identical seeds and
controls, and checks the saved geometry and integration counters across
backends.

```sh
NUMBA_NUM_THREADS=8 python validation/benchmark_threading_layers.py \
  --layers omp tbb --n 64 --threads 2 8 --reference none \
  --out validation/output/threading
```

Timing depends on the seed distribution, storage, workload size, and machine.
