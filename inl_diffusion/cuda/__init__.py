"""
Triton CUDA Kernels for INL-Diffusion

Fused kernels for:
- INL dynamics (5 ops -> 1 kernel)
- MoE expert routing with CGGR (5-6x speedup)
- AdaLN-Zero (for DiT)
- RMSNorm
- SwiGLU

Author: Boris Peyriguere
"""

from .triton_kernels import (
    fused_inl_dynamics,
    fused_inl_dynamics_autograd,
    fused_adaln,
    fused_adaln_gate,
    fused_rmsnorm,
    fused_swiglu,
    # Robotics Control Loop Pattern
    fused_inl_adaln_residual,
    RoboticsDiTLayer,
    HAS_TRITON
)

from .fused_moe_dit import (
    FusedMoEIntegrator,
    AdaptiveMoEIntegrator,
    HAS_FUSED_MOE
)

__all__ = [
    # INL dynamics
    'fused_inl_dynamics',
    'fused_inl_dynamics_autograd',
    # DiT fused ops
    'fused_adaln',
    'fused_adaln_gate',
    'fused_rmsnorm',
    'fused_swiglu',
    # Robotics Control Loop
    'fused_inl_adaln_residual',
    'RoboticsDiTLayer',
    # MoE
    'FusedMoEIntegrator',
    'AdaptiveMoEIntegrator',
    # Flags
    'HAS_TRITON',
    'HAS_FUSED_MOE'
]
