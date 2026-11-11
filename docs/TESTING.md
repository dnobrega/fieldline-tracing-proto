# Testing and reproducibility

From a source checkout, install `python -m pip install -e ".[test,plot]"`
and run `python -m pytest -q`. Tests default to two Numba threads. The suite
covers both methods, nonuniform grids, XYZ memory layouts, periodic seams,
boundary events, storage modes, sampling, diagnostics, and saved results.
One boundary regression builds a 256³ float32 field (about 192 MiB for
three components). The first test run includes Numba compilation. The 0.2.0 suite passed
locally on macOS arm64 with Python 3.14.5, NumPy 2.4.6, and Numba 0.68.1.

The GitHub Actions matrix covers Python 3.10, 3.12 and 3.14 on Linux, plus
Python 3.14 on macOS. It builds distributions and checks a normal wheel
installation outside the checkout. The source checkout includes benchmark
fixtures used by the tests.

After changing numerical helpers, run in a fresh process with a new writable
`NUMBA_CACHE_DIR`. For throughput measurements, warm up a representative
dtype, array layout and set of options. Record package versions, controls,
threading layer, and output sizes alongside timing results. See the
[reproducible studies](../validation/README.md).
