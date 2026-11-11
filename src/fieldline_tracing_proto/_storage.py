"""Bounded optimistic branch buffers, compiled compaction and exact retries."""

from numbers import Integral
import numpy as np
from numba import get_num_threads
from ._kernels import _single_block, _count_lines, _fill_lines, _compact_block


def _starts(counts):
    sizes = counts.sum(axis=1) + 1
    starts = np.empty(len(sizes), dtype=np.int64)
    if len(sizes):
        starts[0] = 0
        np.cumsum(sizes[:-1], out=starts[1:])
    return sizes, starts


def _vectors(total, dtype):
    return [np.empty(total, dtype=dtype) for _ in range(3)] + [
        np.empty(total, dtype=np.float64)
    ]


def _exact(field, seeds, options, dtype, balanced):
    n = len(seeds)
    counts = np.empty((n, 2), dtype=np.int64)
    stats = np.empty((n, 2, 5))
    endpoints = np.empty((n, 2, 3))
    order = np.random.default_rng(0).permutation(n) if balanced else np.arange(n)
    _count_lines(
        *field[:11], *field[11], seeds, options, order, counts, stats, endpoints
    )
    sizes, starts = _starts(counts)
    vectors = _vectors(int(sizes.sum()), dtype)
    ok = np.zeros(n, dtype=np.bool_)
    _fill_lines(
        *field[:11],
        *field[11],
        seeds,
        options,
        order,
        starts,
        counts,
        stats,
        *vectors,
        ok,
    )
    if not np.all(ok):
        raise RuntimeError("Count/fill mismatch; output discarded.")
    return counts, stats, endpoints, vectors, starts


def integrate_storage(
    field,
    seeds,
    options,
    dtype,
    storage,
    memory_limit_mb,
    block_size,
    seed_order,
    buffer_points,
):
    if storage not in ("auto", "single_pass", "two_pass"):
        raise ValueError("Unknown storage mode.")
    if seed_order not in ("balanced", "input"):
        raise ValueError("Unknown seed_order.")
    for name, value in (("buffer_points", buffer_points), ("block_size", block_size)):
        if value is None:
            continue
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(memory_limit_mb) or memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive.")
    n = len(seeds)
    threads = get_num_threads()
    balanced = seed_order == "balanced"
    requested = max(128, 64 * threads) if block_size is None else int(block_size)
    required = (int(options[2]) + int(options[3]) - 1) // int(options[3])
    capacity = min(required, 256 if buffer_points is None else int(buffer_points))
    pointbytes = 3 * dtype.itemsize + 8
    budget = int(memory_limit_mb * 1024**2)
    rowbytes = (2 * capacity + 1) * pointbytes
    minimum = min(n, requested, 16 * threads)
    # RK2's fixed budget is useful when full rows still keep enough workers.
    # Large RK45 safety budgets always start with bounded optimistic capacity.
    if buffer_points is None and options[0] == 0:
        fullbytes = (2 * required + 1) * pointbytes
        if fullbytes * max(1, minimum) <= budget:
            capacity = required
            rowbytes = fullbytes

    if storage == "single_pass" and rowbytes > budget:
        raise ValueError(
            "A branch-buffer row exceeds memory_limit_mb; reduce buffer_points or use storage=auto/two_pass."
        )
    exact = (
        storage == "two_pass"
        or rowbytes > budget
        or (storage == "auto" and budget // rowbytes < minimum)
    )
    info = dict(
        storage="two_pass" if exact else "single_pass",
        num_threads=threads,
        buffer_points=None if buffer_points is None else int(buffer_points),
        block_size=None if block_size is None else int(block_size),
        effective_block_size=0,
        buffer_capacity_peak=0,
        peak_branch_buffer_bytes=0,
        overflow_retraced=0,
        work_nfev=0,
    )
    if not n:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 2, 5)),
            np.empty((0, 2, 3)),
            _vectors(0, dtype),
            info,
        )
    if exact:
        counts, stats, endpoints, vectors, _ = _exact(
            field, seeds, options, dtype, balanced
        )
        info["work_nfev"] = 2 * int(stats[:, :, 4].sum())
        return counts, stats, endpoints, vectors, info
    counts = np.empty((n, 2), dtype=np.int64)
    stats = np.empty((n, 2, 5))
    endpoints = np.empty((n, 2, 3))
    parts = [[], [], [], []]
    begin = 0
    remaining_exact = False
    while begin < n:
        if remaining_exact:
            cc, st, ep, vectors, _ = _exact(
                field, seeds[begin:], options, dtype, balanced
            )
            counts[begin:] = cc
            stats[begin:] = st
            endpoints[begin:] = ep
            for axis in range(4):
                parts[axis].append(vectors[axis])
            info["work_nfev"] += 2 * int(st[:, :, 4].sum())
            info["storage"] = "mixed"
            break
        row_size = 2 * capacity + 1
        rowbytes = row_size * pointbytes
        size = min(n - begin, requested, budget // rowbytes)
        end = begin + size
        info["effective_block_size"] = max(info["effective_block_size"], size)
        info["buffer_capacity_peak"] = max(info["buffer_capacity_peak"], capacity)
        info["peak_branch_buffer_bytes"] = max(
            info["peak_branch_buffer_bytes"], size * rowbytes
        )
        order = (
            np.random.default_rng(begin).permutation(size)
            if balanced
            else np.arange(size)
        )
        buffers = tuple(
            np.empty((size, row_size), dtype=np.float64 if a == 3 else dtype)
            for a in range(4)
        )
        overflow = np.zeros(size, dtype=np.bool_)
        cb = counts[begin:end]
        sb = stats[begin:end]
        eb = endpoints[begin:end]
        _single_block(
            *field[:11],
            *field[11],
            seeds[begin:end],
            options,
            order,
            capacity,
            *buffers,
            cb,
            sb,
            eb,
            overflow,
        )
        info["work_nfev"] += int(sb[:, :, 4].sum())
        ids = np.flatnonzero(overflow)
        retry_map = np.full(size, -1, dtype=np.int64)
        retry_vectors = _vectors(0, dtype)
        retry_starts = np.empty(0, dtype=np.int64)
        if len(ids):
            cc, st, ep, retry_vectors, retry_starts = _exact(
                field, seeds[begin:end][ids], options, dtype, balanced
            )
            cb[ids] = cc
            sb[ids] = st
            eb[ids] = ep
            retry_map[ids] = np.arange(len(ids))
            info["overflow_retraced"] += len(ids)
            info["work_nfev"] += 2 * int(st[:, :, 4].sum())
        sizes, starts = _starts(cb)
        vectors = _vectors(int(sizes.sum()), dtype)
        _compact_block(
            *buffers,
            cb,
            sb,
            capacity,
            starts,
            retry_map,
            retry_starts,
            *retry_vectors,
            *vectors,
        )
        for axis in range(4):
            parts[axis].append(vectors[axis])
        begin = end
        # Learn from common overflows, retaining small buffers for rare outliers.
        if len(ids) > 0.05 * size and begin < n:
            typical = float(np.quantile(cb.max(axis=1), 0.9))
            desired = min(required, max(capacity, int(np.ceil(1.25 * typical))))
            floor = min(n - begin, requested, 16 * threads) if storage == "auto" else 1
            max_capacity = (budget // max(1, floor) // pointbytes - 1) // 2
            if desired > max_capacity and storage == "auto":
                remaining_exact = True
            else:
                capacity = min(desired, max_capacity)
    vectors = [np.concatenate(a) if len(a) > 1 else a[0] for a in parts]
    return counts, stats, endpoints, vectors, info
