"""
INL-DiT: Diffusion Transformer with Integrator Neurons

Architecture based on DiT (Diffusion Transformer) with INL innovations:
- Integrator Neurons for adaptive computation
- MoE (Mixture of Experts) for patch-specific processing
- GQA (Grouped Query Attention)
- RoPE positional encoding for 2D patches
- AdaLN-Zero conditioning
- Triton CUDA kernels for INL dynamics (3x speedup)

Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

# Try to import Triton-accelerated modules
try:
    from ..cuda import fused_inl_dynamics, FusedMoEIntegrator, HAS_TRITON, HAS_FUSED_MOE
except ImportError:
    HAS_TRITON = False
    HAS_FUSED_MOE = False
    fused_inl_dynamics = None
    FusedMoEIntegrator = None


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding for image patches."""

    def __init__(self, dim: int, max_size: int = 64):
        super().__init__()
        self.dim = dim
        self.max_size = max_size

        # Create 2D position embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rotary embeddings for h x w grid."""
        # Create position indices
        y_pos = torch.arange(h, device=device).float()
        x_pos = torch.arange(w, device=device).float()

        # Create 2D grid
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")
        y_grid = y_grid.flatten()
        x_grid = x_grid.flatten()

        # Compute rotary embeddings for both dimensions
        y_emb = torch.outer(y_grid, self.inv_freq)
        x_emb = torch.outer(x_grid, self.inv_freq)

        # Concatenate sin and cos for both dimensions
        emb = torch.cat([y_emb, x_emb], dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to queries and keys."""
    # Split for rotation
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    # Apply rotation
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_rot, k_rot


class IntegratorNeuron(nn.Module):
    """
    Integrator Neuron for adaptive computation in diffusion.

    Allows the model to allocate more computation to complex patches.

    v2 improvements:
    - Per-channel integration weights (not scalar)
    - Triton-accelerated INL dynamics
    - Learned equilibrium point
    - Damped harmonic oscillator dynamics
    """

    def __init__(
        self,
        d_model: int,
        num_iterations: int = 2,
        dt: float = 0.1,
        use_triton: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_iterations = num_iterations
        self.dt = dt
        self.use_triton = use_triton and HAS_TRITON

        # Per-channel integration weights (learned)
        self.integration_weight = nn.Parameter(torch.ones(d_model) * 0.5)

        # Learned equilibrium point
        self.mu = nn.Parameter(torch.zeros(d_model))

        # Halting gate (for adaptive computation)
        self.halt_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Controller - outputs alpha, beta, gate for INL dynamics
        self.controller = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),  # x + v as input
            nn.GELU(),
            nn.Linear(d_model // 2, d_model * 3),  # alpha, beta, gate
        )

        # Refinement layers
        self.refine = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def _compute_dynamics(
        self,
        x: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute INL dynamics with optional Triton acceleration."""
        B, N, D = x.shape

        # Build context
        ctx = torch.cat([x, v], dim=-1)  # [B, N, 2*D]
        ctrl_out = self.controller(ctx)  # [B, N, 3*D]

        # Split into alpha, beta, gate
        alpha_raw, beta_raw, gate_raw = torch.split(ctrl_out, self.d_model, dim=-1)
        alpha = torch.sigmoid(alpha_raw)
        beta = F.softplus(beta_raw)
        gate = torch.sigmoid(gate_raw)

        # Use Triton kernel if available
        if self.use_triton and fused_inl_dynamics is not None and x.is_cuda:
            x_flat = x.view(-1, D)
            v_flat = v.view(-1, D)
            alpha_flat = alpha.view(-1, D)
            beta_flat = beta.view(-1, D)
            gate_flat = gate.view(-1, D)

            x_next_flat, v_next_flat = fused_inl_dynamics(
                x_flat, v_flat, self.mu, alpha_flat, beta_flat, gate_flat, self.dt
            )

            x_next = x_next_flat.view(B, N, D)
            v_next = v_next_flat.view(B, N, D)
        else:
            # PyTorch fallback
            error = x - self.mu
            v_next = alpha * v - beta * error
            x_next = x + self.dt * gate * v_next

        return x_next, v_next

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply integrator neuron processing.

        Args:
            x: Input tensor [B, N, D]
            context: Optional context for integration

        Returns:
            output: Refined tensor [B, N, D]
            aux: Auxiliary info (iterations, halting probs, etc.)
        """
        B, N, D = x.shape

        # Initialize velocity
        v = torch.zeros_like(x)
        integrated = x.clone()

        halt_probs = []
        for iteration in range(self.num_iterations):
            # Compute INL dynamics
            x_next, v_next = self._compute_dynamics(integrated, v)

            # Compute halting probability
            halt_prob = self.halt_gate(x_next)
            halt_probs.append(halt_prob)

            # Refine
            refined = self.refine(x_next)

            # Integrate with per-channel weights
            integrated = integrated + halt_prob * refined * self.integration_weight

            v = v_next

        aux = {
            'halt_probs': torch.stack(halt_probs, dim=1),  # [B, num_iter, N, 1]
            'final_velocity': v,
            'iterations': self.num_iterations
        }

        return integrated, aux


class MoEIntegratorNeuron(nn.Module):
    """
    Mixture of Expert Integrator Neuron.

    Uses multiple expert integrators with learned routing.
    Each expert specializes in different patch types (texture, edges, flat, etc.)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        num_iterations: int = 2,
        dt: float = 0.1,
        use_shared_expert: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.use_moe = HAS_FUSED_MOE and FusedMoEIntegrator is not None

        if self.use_moe:
            # Use optimized MoE implementation
            self.integrator = FusedMoEIntegrator(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                num_iterations=num_iterations,
                dt=dt,
                use_shared_expert=use_shared_expert
            )
        else:
            # Fallback to standard IntegratorNeuron
            self.integrator = IntegratorNeuron(
                d_model=d_model,
                num_iterations=num_iterations,
                dt=dt
            )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward through MoE or standard integrator."""
        if self.use_moe:
            return self.integrator(x, context)
        else:
            output, aux = self.integrator(x, context)
            # Add empty MoE info for compatibility
            aux['router_probs'] = None
            aux['expert_usage'] = {}
            return output, aux


class INLDiTAttention(nn.Module):
    """
    Multi-head attention with GQA for INL-DiT.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        # For GQA: KV heads use same head_dim but fewer heads
        self.kv_dim = self.head_dim * num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads for GQA
        num_rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(num_rep, dim=1)
        v = v.repeat_interleave(num_rep, dim=1)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.out_proj(out)


class INLDiTCrossAttention(nn.Module):
    """Cross-attention for text conditioning."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        num_heads: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(context_dim, d_model, bias=False)
        self.v_proj = nn.Linear(context_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        _, S, _ = context.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.out_proj(out)


class INLDiTMLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class INLDiTBlock(nn.Module):
    """
    INL-DiT Transformer block.

    Components:
    - Self-attention with RoPE
    - Cross-attention for text conditioning
    - MLP
    - Integrator Neuron (standard or MoE)
    - AdaLN-Zero for timestep conditioning
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        context_dim: int = 2048,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_integrator_iterations: int = 2,
        # MoE options
        use_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 2,
    ):
        super().__init__()

        self.use_moe = use_moe

        # Normalization layers
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.norm_context = RMSNorm(context_dim)

        # Attention
        self.attn = INLDiTAttention(d_model, num_heads, num_kv_heads, dropout)
        self.cross_attn = INLDiTCrossAttention(d_model, context_dim, num_heads, dropout)

        # MLP
        self.mlp = INLDiTMLP(d_model, mlp_ratio, dropout)

        # Integrator Neuron (MoE or standard)
        if use_moe:
            self.integrator = MoEIntegratorNeuron(
                d_model=d_model,
                num_experts=num_experts,
                top_k=moe_top_k,
                num_iterations=num_integrator_iterations,
            )
        else:
            self.integrator = IntegratorNeuron(d_model, num_integrator_iterations)

        # AdaLN-Zero modulation parameters (6 for self-attn, cross-attn, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_emb: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            x: Image patch embeddings [B, N, D]
            context: Text embeddings [B, S, context_dim]
            t_emb: Timestep embedding [B, D]
            cos, sin: Rotary position embeddings

        Returns:
            x: Updated patch embeddings [B, N, D]
            aux: Auxiliary info from integrator
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)

        # Self-attention with AdaLN
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm, cos, sin)

        # Cross-attention (text conditioning)
        x = x + self.cross_attn(self.norm2(x), self.norm_context(context))

        # MLP with AdaLN
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        # Integrator Neuron refinement
        x, aux = self.integrator(x, context)

        return x, aux


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.

        Args:
            t: Timestep tensor [B]

        Returns:
            Timestep embedding [B, d_model]
        """
        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return self.mlp(embedding)


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        d_model: int = 1152,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.proj = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed patches.

        Args:
            x: Input latent [B, C, H, W]

        Returns:
            Patch embeddings [B, N, D]
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FinalLayer(nn.Module):
    """Final layer for noise prediction."""

    def __init__(self, d_model: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.linear = nn.Linear(d_model, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class INLDiT(nn.Module):
    """
    INL Diffusion Transformer.

    A transformer-based diffusion model with Integrator Neurons
    for text-to-image generation.

    Architecture:
    - Patch embedding for latent images
    - Timestep embedding
    - INL-DiT blocks with self-attention, cross-attention, MLP, and Integrator Neurons
    - Final layer for noise prediction

    v2 improvements:
    - MoE (Mixture of Experts) for Integrator Neurons
    - Triton CUDA kernels for 3x speedup
    - Proper CFG with learned null embeddings
    """

    # Model configurations
    CONFIGS = {
        "S": {  # Small ~100M params
            "d_model": 384,
            "num_layers": 12,
            "num_heads": 6,
            "num_kv_heads": 2,
        },
        "B": {  # Base ~250M params
            "d_model": 768,
            "num_layers": 12,
            "num_heads": 12,
            "num_kv_heads": 4,
        },
        "L": {  # Large ~500M params
            "d_model": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "num_kv_heads": 4,
        },
        "XL": {  # XL ~700M params
            "d_model": 1152,
            "num_layers": 28,
            "num_heads": 16,
            "num_kv_heads": 4,
        },
        "XXL": {  # XXL ~1.5B params
            "d_model": 1536,
            "num_layers": 32,
            "num_heads": 24,
            "num_kv_heads": 6,
        },
        # MoE variants
        "L-MoE": {  # Large with MoE
            "d_model": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "num_kv_heads": 4,
            "use_moe": True,
            "num_experts": 4,
            "moe_top_k": 2,
        },
        "XL-MoE": {  # XL with MoE
            "d_model": 1152,
            "num_layers": 28,
            "num_heads": 16,
            "num_kv_heads": 4,
            "use_moe": True,
            "num_experts": 4,
            "moe_top_k": 2,
        },
    }

    def __init__(
        self,
        img_size: int = 32,  # Latent size (256/8 = 32)
        patch_size: int = 2,
        in_channels: int = 4,  # VAE latent channels
        d_model: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        context_dim: int = 2048,  # INL-LLM embedding dim
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_integrator_iterations: int = 2,
        # MoE options
        use_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 2,
        # CFG options
        use_learned_null: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.out_channels = in_channels
        self.use_moe = use_moe
        self.use_learned_null = use_learned_null

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, d_model)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        # Timestep embedding
        self.t_embed = TimestepEmbedding(d_model)

        # 2D Rotary position embedding
        self.rope = RotaryPositionEmbedding2D(d_model // num_heads)

        # Learned null embedding for CFG (instead of zeros)
        if use_learned_null:
            self.null_embedding = nn.Parameter(torch.randn(1, 77, context_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            INLDiTBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                context_dim=context_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                num_integrator_iterations=num_integrator_iterations,
                use_moe=use_moe,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
            )
            for _ in range(num_layers)
        ])

        # Final layer
        self.final_layer = FinalLayer(d_model, patch_size, self.out_channels)

        # Print configuration
        backend = "Triton" if HAS_TRITON else "PyTorch"
        moe_str = f" + MoE({num_experts} experts)" if use_moe else ""
        print(f"INL-DiT initialized: {num_layers}L, d={d_model}, [{backend}{moe_str}]")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Zero-out output layers for residual connections
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch predictions back to image.

        Args:
            x: Patch predictions [B, N, patch_size^2 * C]

        Returns:
            Image [B, C, H, W]
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.grid_size

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))

        return imgs

    def get_null_embedding(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get null embedding for CFG (learned or zeros)."""
        if self.use_learned_null:
            return self.null_embedding.expand(batch_size, -1, -1)[:, :seq_len]
        else:
            return torch.zeros(batch_size, seq_len, self.d_model, device=device)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        use_null_context: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for noise prediction.

        Args:
            x: Noisy latent [B, C, H, W]
            t: Timestep [B]
            context: Text embeddings from INL-LLM [B, S, context_dim]
            use_null_context: If True, use null embedding instead of context (for CFG)

        Returns:
            Predicted noise [B, C, H, W]
        """
        B = x.shape[0]

        # Use null embedding for unconditional generation (CFG)
        if use_null_context:
            context = self.get_null_embedding(B, context.shape[1], x.device)

        # Patch embed
        x = self.patch_embed(x)

        # Timestep embedding
        t_emb = self.t_embed(t)

        # Get rotary embeddings
        cos, sin = self.rope(self.grid_size, self.grid_size, x.device)

        # Transformer blocks
        aux_outputs = []
        for block in self.blocks:
            x, aux = block(x, context, t_emb, cos, sin)
            aux_outputs.append(aux)

        # Final layer
        x = self.final_layer(x, t_emb)

        # Unpatchify
        x = self.unpatchify(x)

        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        cfg_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        Forward pass with Classifier-Free Guidance.

        Combines unconditional and conditional predictions.

        Args:
            x: Noisy latent [B, C, H, W]
            t: Timestep [B]
            context: Text embeddings [B, S, context_dim]
            cfg_scale: CFG scale (typically 7.5)

        Returns:
            Predicted noise with CFG applied [B, C, H, W]
        """
        # Unconditional prediction
        noise_uncond = self.forward(x, t, context, use_null_context=True)

        # Conditional prediction
        noise_cond = self.forward(x, t, context, use_null_context=False)

        # CFG interpolation
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

        return noise_pred

    @classmethod
    def from_config(cls, config_name: str, **kwargs) -> "INLDiT":
        """Create model from predefined config."""
        if config_name not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Choose from {list(cls.CONFIGS.keys())}")

        config = cls.CONFIGS[config_name].copy()
        config.update(kwargs)

        return cls(**config)

    def get_num_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
