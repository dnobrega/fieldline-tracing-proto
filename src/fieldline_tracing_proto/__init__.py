"""Cartesian magnetic field-line tracing, sampling and diagnostics."""

from .diagnostics import LineDiagnosticsResult, compute_line_diagnostics
from .grid import PeriodicInferenceWarning
from .sampling import (
    ScalarSampleResult,
    sample_ragged_to_padded,
    sample_scalar_on_lines,
)
from .tracing import (
    TraceResult,
    exit_code_meaning,
    load_trace_npz,
    print_exit_summary,
    ragged_to_padded,
    save_trace_npz,
    trace_field_lines,
)
from .winding import LocalWindingResult, compute_local_winding

__version__ = "0.2.0"

__all__ = [
    "LineDiagnosticsResult",
    "LocalWindingResult",
    "PeriodicInferenceWarning",
    "ScalarSampleResult",
    "TraceResult",
    "compute_line_diagnostics",
    "compute_local_winding",
    "exit_code_meaning",
    "load_trace_npz",
    "print_exit_summary",
    "ragged_to_padded",
    "sample_ragged_to_padded",
    "sample_scalar_on_lines",
    "save_trace_npz",
    "trace_field_lines",
]
