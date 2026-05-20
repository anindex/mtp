"""Model Tensor Planning (MTP).

A sampling-based model predictive control framework that performs high-entropy
control generation using structured tensor sampling over randomized multipartite
graphs with spline interpolation.

See: https://arxiv.org/abs/2505.01059
"""

from mtp.mtp import MTP, MTPInterpolationType, MTPParams

__all__ = ["MTP", "MTPParams", "MTPInterpolationType"]
