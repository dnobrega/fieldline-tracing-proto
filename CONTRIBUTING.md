# Contributing

From a checkout, create an isolated environment and install the development
extras:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,plot]"
python -m pytest -q
```

For a numerical issue, include the grid shape and axes, field dtype and layout,
seed coordinates, periodic flags and periods, integrator controls, and exit
codes. A small analytic or synthetic reproducer is especially helpful. Run
the relevant tests after changing the interpolation or integration contracts.

Use a fresh Python process and a new `NUMBA_CACHE_DIR` after changing numerical
helpers. Numba can reuse a compiled caller after an imported helper changes.
Measure warm performance separately from the first call. For benchmark
instructions see [validation](validation/README.md).

Before packaging, check the wheel from outside the checkout and run the
standalone plotting example. Keep generated reports and local environments
out of commits.
