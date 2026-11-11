# Public API

Import from `fieldline_tracing_proto`.

## Tracing

```python
trace_field_lines(bx, by, bz, xgrid, ygrid, zgrid, seeds, ds, max_steps, **options)
```

Fields have native shape `(nx,ny,nz)`; seed columns are `(x,y,z)`.
`field_order="zyx"` is an explicit deprecated adapter for old cubes.
`dtype=None` inherits common floating field precision (other types become
float64). `array_order="K"` preserves C/F/strided layouts; C/F can be requested.
`field_layout="interleaved"` optionally copies the components into
`(nx,ny,nz,3)` storage and exposes component views to the shared interpolator.
It trades allocation/copy cost for locality; a speedup is workload dependent.

The method defaults to RK2. `ds` is a fixed RK2 step or initial RK45 step.
`max_steps` and `max_length` bound each branch. RK2 rejects explicit adaptive
controls (`rtol`, `atol`, `min_step`, `max_step`, `max_attempts`, `error_scale`),
whose sentinels are None. `max_step_cells` is also adaptive-only.
See [RK45](RK45.md) for adaptive defaults.
`bmin=1e-12` stops normalized tracing at weak fields.

Omitted X/Y flags still resolve to True (Z defaults False), but omission or
period inference emits `PeriodicInferenceWarning`. Set flags and physical
periods explicitly for simulation data, especially cropped subdomains.
`warn_periodic=False` suppresses that warning when the assumptions are known;
metadata still records inferred axes and defaulted flags.
`periodic_convention="endpoint_excluded"` is the default; uniform periodic
axes infer n·dx. `"endpoint_included"` uses last−first for a repeated endpoint.
`period=(Px,Py,Pz)` overrides inferred lengths; disabled axes take 0 or None.
A nonuniform endpoint-excluded periodic axis requires its physical period.
Periodic seeds are wrapped; seeds outside a nonperiodic axis raise ValueError.
`original_seeds` retains supplied positions in float64.

`save_stride=1` saves accepted endpoints, the seed, and the terminal point.
Successive positions identical at output precision are merged only when their
arc separation is consistent with coordinate roundoff. Substantial closed-path
returns (including full wrapped periods) retain their nonzero arc. It is a step stride, not independent uniform physical output spacing.
`wrap_output=False` stores continuous periodic positions (sampling still wraps).

## Storage

`storage="auto"` uses bounded branch buffers and compiled block compaction.
With `buffer_points=None`, RK45 starts with up to 256 saved points per branch,
independent of a large max_steps safety limit. RK2 may reserve its full fixed
budget if the memory bound still accommodates at least the selected minimum
parallel batch; otherwise it also starts optimistically. Explicit positive
`buffer_points` fixes the initial capacity, capped by the step/stride budget.

A full buffer is an **internal retry**, never an interpolation exit. Only seeds
that overflow are retraced using exact count/fill storage. Common overflows
can enlarge later buffers based on the observed 90th-percentile size. Auto mode
uses exact two-pass storage when memory cannot sustain the minimum batch;
a transition after an optimistic block is recorded as storage="mixed".
Explicit "single_pass" retains optimistic blocks/retries and raises if even
one requested row cannot fit. "two_pass" always uses exact count/fill.

`block_size=None` requests max(128,64*Numba threads) seeds per block, subject
to available seeds and `memory_limit_mb=64` MiB. Auto mode uses a minimum
batch heuristic of min(available,requested,16*threads) before preferring exact
storage. Explicit positive block_size overrides the target. These are workload
heuristics, not guarantees of parallel efficiency. `seed_order="balanced"`
uses deterministic permutations, preserving the external seed/result order;
"input" disables this. Independent rows are compacted in compiled code.

The memory limit bounds **branch buffers only**. It excludes inputs, small
workspaces/metadata, exact retry outputs, retained/final outputs, compaction,
concatenation and Python/Numba runtime memory. Large requested output can still
need substantial RAM. Check `peak_branch_buffer_bytes`, not process RSS, to
verify that particular budget.

Settings record actual storage mode, num_threads, block/capacity peaks, overflow_retraced,
and work_nfev (including initial partial integrations and both exact passes).
Logical nfev_minus/plus still describe each final branch integration once.

## TraceResult and persistence

`get_line(i)` returns `(x,y,z,s)` views into concatenated ragged arrays.
`line_start/count` delimit each line. `seed_index` is an absolute array index.
`sfl` is float64 integrated arc from the minus endpoint, not chord length.
`nsteps_minus/plus`, `length_minus/plus`, `nrejected_minus/plus`,
`nfev_minus/plus` describe each branch. In two-pass mode nfev describes one
integration; total integration evaluation work is twice that value.
`endpoints_minus/plus` retain float64 endpoint positions and
`exit_faces_minus/plus` give absolute face codes or 0 for other terminations.
`settings_json` records conventions, layout, actual storage mode and controls,
not the field, grids or data provenance. Preserve those separately.
`ragged_to_padded` returns four `(nlines,max_points)` matrices; s stays float64.

`save_trace_npz(..., compressed=True)` writes schema 3 without pickle.
`load_trace_npz` accepts schemas 2/3 and original files; unavailable fields are
None. Old files retain old geometry and metadata; loading does not retrace them.

| Absolute code | Reason |
| --- | --- |
| 1/2 | zmin/zmax |
| 3/4 | ymin/ymax |
| 5/6 | xmin/xmax |
| 7 | max_steps |
| 8 | weak field |
| 9 | NaN/Inf |
| 10 | interpolation/internal storage error |
| 11 | max_length |
| 12 | step underflow/no representable progress |
| 13 | max_attempts |

Minus-branch codes are negative, plus-branch positive. Face metadata is unsigned.

## Sampling and diagnostics

`sample_scalar_on_lines(result, scalar, xgrid, ygrid, zgrid, ...)` accepts
native XYZ scalar cubes and preserves their C/F/strided layout. Flags,
periods and output dtype default to trace metadata/precision. Explicit flag
or period disagreement warns. Legacy metadata without periods uses the old
repeated-endpoint convention. Work positions/grids stay float64. Invalid
samples use `bad_value=np.nan` and a nonzero `codes_all` entry.
`exact_anchors=True` restores float64 seed/endpoint metadata when the float32
stored coordinate still equals that metadata rounded to float32. This prevents
an exact face such as z=1.1 rounding just outside the scalar domain and losing
a valid terminal interval. Edited coordinates are respected; trace arrays are
never modified. `exact_anchors=False` samples literal stored positions instead.
Only available anchors are recovered; interior trajectory storage precision is
not reconstructed. Exact-face claims refer to float64 endpoint metadata;
saved float32 coordinates retain their output rounding.

`ScalarSampleResult.get_values/get_codes` return per-line views.

`compute_line_diagnostics(result, samples, dtype=np.float64, eps=1e-30)`
uses valid adjacent samples and trapezoidal arc integration. Absolute-weight
percentiles invert the piecewise-linear |f| density inside each valid segment;
they are continuous within segments, but still depend on resolved samples.
No-valid-interval results are NaN. Sparse storage can miss narrow peaks.

## Local winding

`compute_local_winding(result, seed_shape, nsample=200, min_sep=1e-8,
dtype=np.float64, parameterization="seed_arc", plane="mean_tangent", ...)`
compares requested forward neighbors on a two-dimensional seed grid.
`seed_shape` is `(nrows,ncols)` and offsets are `(drow,dcol)`.

Default correspondence uses signed arc relative to each seed, only on their
common interval. The fixed pair plane is normal to the mean of their seed
tangents; its orientation follows increasing arc. It is not a moving frame
along the field, a fan/spine classification or a Gauss linking invariant.
`parameterization="normalized", plane="xy"` explicitly selects the historic
s/L, XY proxy. That mode resamples each line once and caches the geometry;
seed_arc uses two-pointer interpolation over each pair's overlap.

Finite coordinates, strictly increasing finite arc and at least two points
are required; invalid lines/pairs produce no contribution (NaN and n_pairs=0
when none remain). A close projected encounter invalidates the whole pair.
Periodic paths use a continuous cover anchored to minimum-image seed separation.
Wrapped data with saved arc gaps >=half any period are rejected as ambiguous;
use dense storage or `wrap_output=False`. The default sparse_policy="raise"
preserves conservative failure of the whole call. Optional sparse_policy="mask"
excludes only ambiguous lines and warns once with their count. `valid_lines`
and `invalid_reason` expose per-line status: 0 valid, 1 short/nonfinite/
nonmonotone, 2 ambiguous wrapped sampling, 3 degenerate tangent. Disjoint ragged
segments are required for parallel preparation. A valid line can still have
invalid pairs (e.g. close projected encounters). Other sparse sampling can alias
turns. Converge geometry density and nsample separately.


## Scalar controls and boundary details

`ds` accepts finite positive real scalars and zero-dimensional NumPy arrays containing such scalars. `max_attempts` is a positive integer at most `2**53`.
Very large positive save_stride remains valid and saves only terminal points
when it exceeds max_steps. Its requested value is retained in metadata.

Outward-curving tangent boundary starts return the face with zero arc. Flat and inward tangencies can proceed. A face/length tie within eight float64 eps relative to `max_length` receives code 11. See [boundary events](RK45.md#events) for the numerical limits.
