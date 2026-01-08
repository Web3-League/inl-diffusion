"""
INL-Diffusion: Image Generation with Integrator Neurons

A diffusion model for text-to-image generation using INL architecture.

Components:
- VAE: Custom Variational Autoencoder for image tokenization
- INL-DiT: Diffusion Transformer with Integrator Neurons
- Pipeline: Text-to-image generation pipeline
"""

__version__ = "0.1.0"

from .vae import INLVAE
from .dit import INLDiT
from .pipeline import INLDiffusionPipeline

__all__ = ["INLVAE", "INLDiT", "INLDiffusionPipeline"]
