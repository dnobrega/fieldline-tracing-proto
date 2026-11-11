# High-resolution tracing on the Olshevsky 14-null field

This repository-only benchmark samples the analytic field used by nullfinder-proto.
It does not add SciPy or nullfinder to the tracing package's runtime dependencies.
Run from the source checkout with NumPy, Numba and SciPy installed.

## Reproduce

A modest run, including direct analytic trajectory references:

```sh
NUMBA_NUM_THREADS=2 python validation/benchmark_olshevsky.py \
  --n 64 128 256 --threads 2 --out validation/output/olshevsky
```

Check layouts, thread counts and both reference fields on a smaller workload:

```sh
NUMBA_NUM_THREADS=2 python validation/benchmark_olshevsky.py \
  --n 64 128 --orders F C --threads 1 2 --seeds 64 --near-per-null 4 \
  --length 0.5 --reference both --reference-seeds 2 --repeats 3 \
  --out validation/output/olshevsky_smoke
```

A larger performance experiment (references disabled for throughput only):

```sh
NUMBA_NUM_THREADS=16 python validation/benchmark_olshevsky.py \
  --n 256 512 --threads 1 2 4 8 16 --seeds 4096 --near-per-null 64 \
  --length 2 --reference none --out validation/output/olshevsky_large
```

For accuracy/convergence runs use `--reference both`, a longer branch length,
more `--reference-seeds`, and tighter reference settings as appropriate. Vary
`--ds`, `--rtol`, `--atol` and `--max-step-cells` independently of grid resolution.
`--reference-max-step 0.005` halves the default independent-reference step cap.
Default DOP853 tolerances are rtol=1e-10, atol=1e-12; reference error must itself
be converged before quoting small trajectory differences. Trilinear fields have
nonsmooth derivatives at cell faces; the sampled reference is not an exact oracle.

Each grid/dtype/layout runs in a fresh process. Reports are written after each
completed worker. Existing JSON outputs are never overwritten: choose a fresh
`--out` directory. Timing ratios are not enforced by tests.

## Memory and layout

The three XYZ fields are filled directly in the requested C/F order, with no
full meshgrid, intermediate full-size float64 cube, or layout conversion.
The 1D coordinate/Gaussian factors use float64. Approximate component storage:

| N | float32, GiB | float64, GiB |
| --- | --- | --- |
| 128 | 0.0234 | 0.0469 |
| 256 | 0.1875 | 0.3750 |
| 512 | 1.5 | 3 |
| 1024 | 12 | 24 |

`--max-field-gib` defaults to 4 and refuses larger **field** allocations before
generating them. A 1024³ float32 run requires explicitly passing at least
`--max-field-gib 12`; float64 needs at least 24. This is not a total-memory bound
or an automatic check of available RAM. Outputs, buffers, imports, JIT and
reference integration require additional memory. Do not run large experiments
concurrently. `--memory-limit-mb` independently caps tracer branch buffers only.

Peak RSS is cumulative within each worker. Later cases can inherit peaks from
previous solves and references. Baseline, post-generation, through-solve and
including-reference peaks are recorded separately; differences are not an
isolated measurement of solver allocations. macOS/Linux RSS units are normalized.

## Scientific and timing contracts

- General seeds use a fixed RNG seed; near-null seeds use a deterministic sphere
  about each of the 14 analytically refined roots. The default radius is 0.02 in
  coordinate units, independent of grid resolution. Seeds and hashes are recorded.
  Root refinement uses the analytic formula, **not** the sampled field/nullfinder;
  a near-null seed on a coarse grid need not be close to that grid's own null.
- The tabulated positions are rounded. Refinement checks residuals below 1e-10
  and rejects duplicate roots. Seeds never start exactly on a root.
- All boundaries are explicitly nonperiodic. Periodicity of the background does
  not establish periodic continuity of the complete Gaussian-perturbed field.
- Default branch length is 1. Default RK2 step and RK45 initial step are 0.2 times
  the grid spacing. RK45 max_step is one spacing by default, rtol=1e-7, atol=1e-10.
  These are explicit comparison controls, **not matched-error calibrations**.
  Changing N also changes these default steps. Pass `--ds` to fix the initial/RK2
  step in physical units; the RK45 cap remains controlled separately.
- The generator's first call is separated from allocation/full field generation.
  Every solver case has an untimed first call, which may compile or load caches;
  it is not a controlled cold-compilation measurement. Warm solve timings include
  validation, buffers, overflow retries, compaction and result construction.
  Input construction, garbage collection, verification and references are excluded.
- Cases use a deterministic shuffled order. Repeated calls must reproduce branch
  codes, accepted steps, lengths and saved-point counts. Finite output is checked.
  Methods need not return identical paths/point counts. Thread/layout comparisons
  are measurements, not universal speedups or cluster-scaling evidence.
- `--reference analytic` integrates the formula directly using SciPy DOP853.
  `--reference both` also integrates SciPy's trilinear sampled field. Differences
  are evaluated at each selected trace's saved arc positions, on both branches.
  The analytic comparison combines interpolation/integration/output-rounding
  effects; the sampled comparison estimates integration/output-rounding effects.
  Their errors cannot simply be subtracted to isolate interpolation error.
- References cover selected seeds only and stop at the trace's branch arc length.
  They do not independently validate face-event classification or prove global
  connectivity, winding accuracy, null topology or accuracy of all seeds.
  Near separatrices, tiny perturbations can produce large position differences.
- JSON records software/platform/source hashes, controls, field-generation times,
  actual samples, nfev/work_nfev, rejections, overflow retries, exit histograms,
  steps and output sizes. Inspect termination codes before comparing timings:
  a prematurely stopped trajectory is not a faster solution to the same problem.

## Fixture provenance

`field.py` was copied from
`nullfinder-proto/benchmarks/olshevsky2020_field.py` on 2026-11-09.
It implements Eq. A.1, Table A.1 and the rounded Table A.2 positions described
there as Olshevsky et al. (2020), Appendix A. Its two documented corrections
(Y2=0.117 and the By constant sqrt(30)/7) are retained. The only compatibility
edit removes `zip(strict=True)` after the existing exact-length validation, so
preallocated filling also works on Python 3.9. The upstream BSD-3-Clause license
and copyright are preserved in [LICENSE.md](LICENSE.md). `provenance.json`
records original/local file hashes. The direct point evaluator uses the same
formula/parameters in a separate implementation; it is independent of gridded
interpolation, but is not an independent verification of the published formula.
