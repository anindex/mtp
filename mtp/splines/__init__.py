"""MTP spline interpolation utilities.

Provides Akima and B-spline interpolation for tensor-sampled control paths.
These are used internally by MTP for smooth control trajectory generation
from sparse graph-sampled control waypoints.
"""

from mtp.splines.akima import poly_akima, poly_interpolation
from mtp.splines.bsplines import compute_b_spline_matrix

__all__ = ["poly_akima", "poly_interpolation", "compute_b_spline_matrix"]
