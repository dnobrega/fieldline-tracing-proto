# Changelog

## 0.2.0

First public version of Fieldline Tracing Proto.

- Bidirectional RK2 and adaptive RK45 on rectilinear Cartesian grids.
- Native XYZ field layout, C and Fortran arrays, periodic seams, and exact
  projection of resolved endpoints onto open faces.
- Boundary handling for tangent seeds and shallow returns, ragged traces,
  scalar sampling, line diagnostics, and relative winding.
- An RK2/RK45 comparison example and analytic validation scripts, including
  the Olshevsky et al. (2020) 14-null field.
