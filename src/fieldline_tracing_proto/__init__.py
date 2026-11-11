"""Cartesian magnetic field-line tracing, sampling and diagnostics."""

from .grid import PeriodicInferenceWarning
from .tracing import (
    TraceResult,
    trace_field_lines,
    ragged_to_padded,
    save_trace_npz,
    load_trace_npz,
    print_exit_summary,
    exit_code_meaning,
)
from .sampling import (
    ScalarSampleResult,
    sample_scalar_on_lines,
    sample_ragged_to_padded,
)
from .diagnostics import LineDiagnosticsResult, compute_line_diagnostics
from .winding import LocalWindingResult, compute_local_winding

__version__ = "0.2.0"

__all__ = [
    "PeriodicInferenceWarning",
    "TraceResult",
    "trace_field_lines",
    "ragged_to_padded",
    "save_trace_npz",
    "load_trace_npz",
    "print_exit_summary",
    "exit_code_meaning",
    "ScalarSampleResult",
    "sample_scalar_on_lines",
    "sample_ragged_to_padded",
    "LineDiagnosticsResult",
    "compute_line_diagnostics",
    "LocalWindingResult",
    "compute_local_winding",
]
