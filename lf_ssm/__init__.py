"""
LF-SSM: Lightweight HiPPO-Free State Space Model for Real-Time UAV Tracking.

Paper: "LF-SSM: Lightweight HiPPO-Free State Space Model for Real-Time UAV Tracking"
       Wang et al., Drones 2026, 10, 102.

This package implements the LF-SSM model which reformulates state evolution on
Riemannian manifolds using geodesic dynamics on the unit sphere, eliminating
HiPPO-derived state transition matrices and specialized hardware kernels.
"""

from lf_ssm.config import ModelConfig, LF_SSM_S, LF_SSM_M, LF_SSM_L, LF_SSM_Nano
from lf_ssm.lf_ssm_model import LFSSM

__all__ = [
    "ModelConfig",
    "LF_SSM_S",
    "LF_SSM_M",
    "LF_SSM_L",
    "LF_SSM_Nano",
    "LFSSM",
]
