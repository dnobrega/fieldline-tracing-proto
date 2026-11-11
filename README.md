# Fieldline Tracing Proto

Trace magnetic field lines through three-dimensional Cartesian data. 
The package supports fixed-step midpoint RK2 and adaptive Dormand–Prince
RK45, nonuniform coordinate axes, periodic boundaries, and explicit exits
at open faces. It also provides scalar sampling and diagnostics along traces.

**Author:** Daniel Nóbrega-Siverio (Université Paris-Saclay, CEA, AIM; Rosseland
Centre for Solar Physics, University of Oslo).

**Contributors:**

- Fernando Moreno-Insertis (Instituto de Astrofísica de Canarias; Universidad de
  La Laguna).
- Claude Opus 5.5 (Anthropic), AI assistant: code review, bug fixes, testing,
  packaging and documentation.
- ChatGPT (OpenAI), AI assistant: algorithm development, numerical optimization,
  code review and documentation.

## License and citation

fieldline-tracing-proto is released under the **[BSD 3-Clause License](LICENSE)**.

If you use fieldline-tracing-proto in scientific research, please cite the repository. Citation information is available in [`CITATION.cff`](CITATION.cff).

## Install

Install directly from GitHub:
```bash
pip install git+https://github.com/dnobrega/fieldline-tracing-proto.git
```
Or clone the repository:
```bash
git clone https://github.com/dnobrega/fieldline-tracing-proto.git
cd nullfinder-proto
pip install -e .
```
Requires Python ≥ 3.10.

## Quick start

Pass colocated component arrays in `(nx, ny, nz)` order. Grid coordinates and
seed columns use `(x, y, z)`. Both C and Fortran layouts are supported.

```python
import numpy as np
from fieldline_tracing_proto import trace_field_lines

x = np.linspace(-1.5, 1.5, 101)
y = np.array([-0.5, 0.5])
z = 1.4 * np.linspace(0, 1, 81) ** 1.4
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
decay = np.exp(-Z)
bx, by, bz = np.cos(X) * decay, np.zeros_like(Y), -np.sin(X) * decay


result = trace_field_lines(
    bx, by, bz, x, y, z, seeds=[[0.0, 0.0, 0.8]],
    ds=0.01, max_steps=4000, method="rk45",
    periodic_x=False, periodic_y=False, periodic_z=False,
)

xline, yline, zline, arc = result.get_line(0)
print(result.exit_code_minus[0], result.exit_code_plus[0])  # -1, 1: lower z face
```

Each seed is traced in both directions. RK45 caps steps by the local mesh
spacing by default. Use `method="rk2"` for fixed-step integration. 

## Conventions (please read)

**Co-located data.** The three magnetic-field components `bx`, `by`, and `bz`
must be sampled at the **same grid nodes** `(x[i], y[j], z[k])`.
The caller is responsible for supplying co-located field components and
coordinates consistent with their locations.

**Indexing.** Arrays have shape `(nx, ny, nz)` and are accessed as
`arr[i, j, k]`. Coordinates are 1D, finite and strictly monotonic.

## Warnings

- The tracer integrates the direction of the trilinearly interpolated field.
Results depend on mesh resolution, seed placement, and integration controls.
Check convergence for shallow contacts, near magnetic nulls and separatrices,
and when interpreting connectivity. The field direction is undefined at a
null. Use nearby seeds for neighborhood studies.

- `compute_local_winding` estimates relative rotation under a chosen line
correspondence and frame. Treat it as a proxy. Physical torsion requires
an independently justified definition. The validation studies use analytic
and synthetic fields. 

## Documentation

- [API, inputs, exit codes, and saved results](docs/API.md)
- [RK45 error control and boundary events](docs/RK45.md)
- [Threading configuration](docs/THREADING.md)
- [Tests and reproducible benchmarks](docs/TESTING.md)

## Acknowledgements

This research has been supported by the European Research Council through the
Synergy Grant number 810218 ("The Whole Sun", ERC-2018-SyG); by the Centre
National d'Etudes Spatiales (CNES) Solar Orbiter project; by the Research
Council of Norway through its Centres of Excellence scheme, project number
262622; and by grant CEX2025-001609-S, awarded to the Instituto de Astrofísica
de Canarias under the Severo Ochoa Centre of Excellence program and funded by
MICIU/AEI/10.13039/501100011033.




