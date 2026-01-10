"""
Triton CUDA Kernels for INL-Diffusion

Fused kernels for:
- INL dynamics (5 ops -> 1 kernel)
- MoE expert routing (batched computation)
- Attention optimizations

Author: Boris Peyriguere
"""

from .triton_kernels import (
    fused_inl_dynamics,
    fused_inl_dynamics_autograd,
    HAS_TRITON
)

from .fused_moe_dit import (
    FusedMoEIntegrator,
    HAS_FUSED_MOE
)

__all__ = [
    'fused_inl_dynamics',
    'fused_inl_dynamics_autograd',
    'FusedMoEIntegrator',
    'HAS_TRITON',
    'HAS_FUSED_MOE'
]
