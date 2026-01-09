"""
INL-VAE: Custom Variational Autoencoder for Image Tokenization

Architecture:
- Encoder: Image (H,W,3) → Latent (h,w,latent_dim)
- Decoder: Latent (h,w,latent_dim) → Image (H,W,3)

Features:
- Fully custom, no dependency on external VAE
- KL-regularized latent space
- Residual blocks with group normalization
- Attention at bottleneck resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from pathlib import Path
import json
import math


class GroupNorm32(nn.GroupNorm):
    """Group normalization with 32 groups."""
    def __init__(self, num_channels: int):
        super().__init__(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Swish(nn.Module):
    """Swish activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and Swish activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.norm1 = GroupNorm32(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = GroupNorm32(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.activation = Swish()

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.norm1(x))
        h = self.conv1(h)
        h = self.activation(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for VAE bottleneck."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = GroupNorm32(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)

        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).contiguous().view(B, C, H, W)

        return x + self.proj(out)


class Downsample(nn.Module):
    """2x downsampling with strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x upsampling with nearest neighbor + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class INLVAEEncoder(nn.Module):
    """
    VAE Encoder: Image → Latent

    Downsamples image by factor of 8 (256x256 → 32x32)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_dim: int = 4,
        use_attention: bool = True,
        attention_resolution: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()

        in_ch = base_channels
        current_resolution = 256  # Assuming 256x256 input

        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch

                # Add attention at specified resolution
                if use_attention and current_resolution == attention_resolution:
                    self.down_blocks.append(AttentionBlock(out_ch))

            # Downsample (except last level)
            if i < len(channel_mult) - 1:
                self.down_blocks.append(Downsample(out_ch))
                current_resolution //= 2

        # Middle blocks
        self.mid_block1 = ResidualBlock(in_ch, in_ch)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch)

        # Output
        self.norm_out = GroupNorm32(in_ch)
        self.activation = Swish()

        # Output to latent (mean and logvar)
        self.conv_out = nn.Conv2d(in_ch, latent_dim * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            mean: Latent mean [B, latent_dim, h, w]
            logvar: Latent log variance [B, latent_dim, h, w]
        """
        h = self.conv_in(x)

        for block in self.down_blocks:
            h = block(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.activation(self.norm_out(h))
        h = self.conv_out(h)

        mean, logvar = h.chunk(2, dim=1)

        return mean, logvar


class INLVAEDecoder(nn.Module):
    """
    VAE Decoder: Latent → Image

    Upsamples latent by factor of 8 (32x32 → 256x256)
    """

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_dim: int = 4,
        use_attention: bool = True,
        attention_resolution: int = 32,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim

        # Compute final encoder channels
        final_ch = base_channels * channel_mult[-1]

        # Input from latent
        self.conv_in = nn.Conv2d(latent_dim, final_ch, kernel_size=3, padding=1)

        # Middle blocks
        self.mid_block1 = ResidualBlock(final_ch, final_ch)
        self.mid_attn = AttentionBlock(final_ch)
        self.mid_block2 = ResidualBlock(final_ch, final_ch)

        # Upsampling blocks (reverse order)
        self.up_blocks = nn.ModuleList()

        in_ch = final_ch
        current_resolution = 256 // (2 ** (len(channel_mult) - 1))  # Start at bottleneck resolution

        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = base_channels * mult

            for j in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch

                # Add attention at specified resolution
                if use_attention and current_resolution == attention_resolution:
                    self.up_blocks.append(AttentionBlock(out_ch))

            # Upsample (except last level)
            if i < len(channel_mult) - 1:
                self.up_blocks.append(Upsample(out_ch))
                current_resolution *= 2

        # Output
        self.norm_out = GroupNorm32(in_ch)
        self.activation = Swish()
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.

        Args:
            z: Latent tensor [B, latent_dim, h, w]

        Returns:
            Image reconstruction [B, 3, H, W]
        """
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for block in self.up_blocks:
            h = block(h)

        h = self.activation(self.norm_out(h))
        h = self.conv_out(h)

        return h


class INLVAE(nn.Module):
    """
    INL Variational Autoencoder for image tokenization.

    Encodes images to a continuous latent space for diffusion training.

    Architecture:
    - Encoder: Conv → ResBlocks + Downsample → Attention → μ, logσ²
    - Decoder: Conv → ResBlocks + Upsample → Attention → Image

    Features:
    - 8x spatial compression (256x256 → 32x32)
    - 4-channel latent space (like SD)
    - KL-regularized for smooth latent space
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_dim: int = 4,
        use_attention: bool = True,
    ):
        super().__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.latent_size = image_size // (2 ** (len(channel_mult) - 1))

        # Attention at 32x32 resolution
        attention_resolution = 32

        self.encoder = INLVAEEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            latent_dim=latent_dim,
            use_attention=use_attention,
            attention_resolution=attention_resolution,
        )

        self.decoder = INLVAEDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            latent_dim=latent_dim,
            use_attention=use_attention,
            attention_resolution=attention_resolution,
        )

        # Scaling factor for latent space (helps with diffusion training)
        self.register_buffer("latent_scale", torch.tensor(0.18215))

    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            x: Input image [B, 3, H, W] in range [-1, 1]
            sample: If True, sample from distribution. If False, return mean.

        Returns:
            Latent tensor [B, latent_dim, h, w]
        """
        mean, logvar = self.encoder(x)

        if sample:
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            z = mean

        # Scale latents
        z = z * self.latent_scale

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.

        Args:
            z: Latent tensor [B, latent_dim, h, w]

        Returns:
            Reconstructed image [B, 3, H, W] in range [-1, 1]
        """
        # Unscale latents
        z = z / self.latent_scale

        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode and decode.

        Args:
            x: Input image [B, 3, H, W]
            return_dict: If True, return dict with all outputs

        Returns:
            recon: Reconstructed image
            mean: Latent mean
            logvar: Latent log variance
        """
        mean, logvar = self.encoder(x)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        # Decode
        recon = self.decoder(z)

        if return_dict:
            return {
                "reconstruction": recon,
                "mean": mean,
                "logvar": logvar,
                "latent": z,
            }

        return recon, mean, logvar

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1e-6,
    ) -> dict:
        """
        Compute VAE loss.

        Args:
            x: Original image
            recon: Reconstructed image
            mean: Latent mean
            logvar: Latent log variance
            kl_weight: Weight for KL divergence term

        Returns:
            Dict with loss components
        """
        # Reconstruction loss (MSE or L1)
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def get_num_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def tokenize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Tokenize images to latents (for diffusion training).

        Args:
            images: Batch of images [B, 3, H, W] in range [0, 1]

        Returns:
            Latents [B, latent_dim, h, w]
        """
        # Normalize to [-1, 1]
        x = images * 2 - 1
        return self.encode(x, sample=False)

    @torch.no_grad()
    def detokenize(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Detokenize latents back to images.

        Args:
            latents: Latent tensor [B, latent_dim, h, w]

        Returns:
            Images [B, 3, H, W] in range [0, 1]
        """
        x = self.decode(latents)
        # Denormalize from [-1, 1] to [0, 1]
        return (x + 1) / 2

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save model to directory with config and weights."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "image_size": self.image_size,
            "latent_dim": self.latent_dim,
            "base_channels": self.encoder.base_channels,
            "in_channels": self.encoder.in_channels,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save weights as safetensors
        try:
            from safetensors.torch import save_file
            save_file(self.state_dict(), save_dir / "model.safetensors")
        except ImportError:
            torch.save(self.state_dict(), save_dir / "model.pt")

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path], device: str = "cpu") -> "INLVAE":
        """Load model from directory or checkpoint file."""
        model_path = Path(model_path)

        # Load from directory
        if model_path.is_dir():
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            else:
                config = {}

            # Create model
            model = cls(
                image_size=config.get("image_size", 256),
                latent_dim=config.get("latent_dim", 4),
                base_channels=config.get("base_channels", 128),
                in_channels=config.get("in_channels", 3),
            )

            # Load weights
            safetensors_path = model_path / "model.safetensors"
            pt_path = model_path / "model.pt"

            if safetensors_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path, device=device)
            elif pt_path.exists():
                state_dict = torch.load(pt_path, map_location=device)
            else:
                raise FileNotFoundError(f"No model weights found in {model_path}")

            model.load_state_dict(state_dict)

        # Load from .pt checkpoint file
        elif model_path.suffix == ".pt":
            checkpoint = torch.load(model_path, map_location=device)
            if "config" in checkpoint:
                config = checkpoint["config"]
            else:
                config = {}

            model = cls(
                image_size=config.get("image_size", 256),
                latent_dim=config.get("latent_dim", 4),
                base_channels=config.get("base_channels", 128),
            )

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

        else:
            raise ValueError(f"Unsupported model path: {model_path}")

        return model
