# Fieldline Tracing Proto

Trace magnetic field lines through three-dimensional Cartesian simulation
data. The package supports fixed-step midpoint RK2 and adaptive Dormand–Prince
RK45, nonuniform coordinate axes, periodic boundaries, and explicit exits
at open faces. It also provides scalar sampling and diagnostics along traces.

![Magnetic field lines traced through a potential arcade. The seed points lie along the central vertical axis and each loop ends at the lower face.](docs/images/arcade.png)

## Install

Use Python 3.10 or later. From a local checkout or an extracted source archive:

```sh
python -m pip install .
```

Install with `python -m pip install ".[plot]"` to run the plotting examples.

## Trace a field line

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
spacing by default. Use `method="rk2"` for fixed-step integration. Run
`python examples/plot_arcade.py` to reproduce the figure.

For Bifrost, pass the centered component cubes and their native physical axes.
Set periodic flags explicitly and supply the physical periods for periodic
axes. See the [Bifrost workflow](docs/BIFROST.md) for sampling near nulls.

## Documentation

- [API, inputs, exit codes, and saved results](docs/API.md)
- [RK45 error control and boundary events](docs/RK45.md)
- [Threading configuration](docs/THREADING.md)
- [Tests and reproducible benchmarks](docs/TESTING.md)

## Numerical limits

The tracer integrates the direction of the trilinearly interpolated field.
Results depend on mesh resolution, seed placement, and integration controls.
Check convergence for shallow contacts, near magnetic nulls and separatrices,
and when interpreting connectivity. The field direction is undefined at a
null. Use nearby seeds for neighborhood studies.

`compute_local_winding` estimates relative rotation under a chosen line
correspondence and frame. Treat it as a proxy. Physical torsion requires
an independently justified definition. The validation studies use analytic
and synthetic fields. A comparison on a real Bifrost snapshot and cluster
scaling remain open.

## Citation and license

Cite the version you used with the metadata in [CITATION.cff](CITATION.cff).
The code is licensed under [BSD-3-Clause](LICENSE). Development used ChatGPT
assistance.
