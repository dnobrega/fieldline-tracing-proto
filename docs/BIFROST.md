# Bifrost and magnetic-null workflow

Create BifrostData once and reuse it. The reader and null finder are separate
projects; this package neither reads their simulation files nor bundles them.
Their public centered cubes use `(nx,ny,nz)` with physical `(x,y,z)` coordinates.
The tracer and scalar sampler use the same `(nx,ny,nz)` order. C and Fortran
contiguous cubes can both be passed directly.

```python
from bifrost_proto import BifrostData
from fieldline_tracing_proto import trace_field_lines
import nullfinder_proto as nf

dd = BifrostData()
snap = dd.snaps[-1]
xx, yy, zz = dd.get_x(), dd.get_y(), dd.get_z()
bx = dd.get_var("b_x", snap, center=True)
by = dd.get_var("b_y", snap, center=True)
bz = dd.get_var("b_z", snap, center=True)

# Keep original axes for nullfinder_proto.
# In this reader z= selects physical Mm; iz= selects public grid indices.
zsub = dd.get_z(z=slice(4,20))
bsub = [dd.get_var(name, snap, z=slice(4,20), center=True)
        for name in ("b_x", "b_y", "b_z")]
res = nf.find_magnetic_nulls(*bsub, xx, yy, zsub, classify=True)

# Keep these XYZ cubes for tracing and scalar sampling.
```

## Confirm that both routines see the same null

```python
import numpy as np
from scipy.interpolate import RegularGridInterpolator

nulls = np.asarray(res["pos"], dtype=np.float64)
values = np.column_stack([
    RegularGridInterpolator((xx,yy,zz), a)(nulls) for a in (bx,by,bz)
])
print("Independent residual:", np.linalg.norm(values,axis=1))
print("Null-finder residual:", res["bnorm"])
```

This checks the same XYZ arrays used by the tracer. Never swap vector
components or reverse query columns. Equal X/Z sizes can conceal an axis mismatch.
If residuals still disagree, compare the selected centered subvolume with the
same region of the centered full cube before interpreting trajectories.

## Seeds around a null

The normalized field direction is undefined at an exact null. With residual
below bmin, the tracer returns only the seed and exit codes (-8,8). To explore
the neighborhood, use displaced seeds; these are not automatically the spine
or fan, which require the Jacobian's local invariant directions/subspace.

```python
null = nulls[0]
widths = []
for grid, value in zip((xx,yy,zz), null):
    j = np.clip(np.searchsorted(grid,value,side="right")-1, 0, len(grid)-2)
    widths.append(grid[j+1]-grid[j])
radius = 0.05*min(widths)  # Initial trial radius; test convergence.
i = np.arange(64)
u = 1-2*(i+0.5)/len(i)
phi = np.pi*(3-np.sqrt(5))*i
r = np.sqrt(1-u*u)
seeds = null + radius*np.column_stack((r*np.cos(phi),r*np.sin(phi),u))

spacing = min(np.min(np.diff(g)) for g in (xx,yy,zz))
result = trace_field_lines(
    bx, by, bz, xx, yy, zz, seeds,
    ds=0.2*spacing, max_steps=20000, method="rk45",
    max_length=2*max(np.ptp(g) for g in (xx,yy,zz)),
    rtol=1e-7, atol=1e-9*spacing, dtype=np.float64,
    periodic_x=True, periodic_y=True, periodic_z=False,
)  # Default max_step limits RK45 to local mesh spacing.

```

Ensure displaced seeds are in the intended domain and above the weak-field
threshold. Compare radii and integration controls. Pass the original XYZ arrays
to trace_field_lines, with coordinate and seed order unchanged. Use float64
for initial precision studies; do not infer fan/spine geometry from arbitrary
seed directions or a single two-dimensional projection.

## XZ quick look near a null

This workflow assumes that you already have the Bifrost reader and null finder.
The [standalone arcade example](../examples/plot_arcade.py) runs on synthetic data.
Install plotting dependencies with `python -m pip install -e ".[plot]"`.
In a notebook whose folder contains the extracted project:

```python
%run examples/plot_bifrost_xz.py
fig, ax = plot_bifrost_xz(dd, snap, res["pos"][0], half_width=3., density=1.5)
plt.show()
```

Alternatively give `%run` the absolute path to the example. It selects the
nearest Y plane and plots |B| plus 2-D streamlines of (Bx,Bz), with the null's
XZ projection marked. Both vector components are centered and remeshed onto
one uniform plotting grid. These curves are not 3-D RK trajectories when
By is nonzero. The example was visually checked on synthetic nonuniform-grid
data; it does not constitute validation of your simulation reader/data.
