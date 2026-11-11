import os

# Small deterministic fixtures do not benefit from using all CPU cores.
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
