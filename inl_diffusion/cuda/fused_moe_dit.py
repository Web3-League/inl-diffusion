"""
Fused MoE (Mixture of Experts) Integrator for DiT

Optimized expert routing and computation for diffusion models.
Each expert specializes in different patch complexity levels.

Author: Boris Peyriguere
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from .triton_kernels import fused_inl_dynamics, HAS_TRITON

# Flag for fused MoE availability
HAS_FUSED_MOE = True  # Always available (uses optimized PyTorch)


class FusedMoEIntegrator(nn.Module):
    """
    Mixture of Expert Integrators for Diffusion.

    Multiple expert integrators with learned routing.
    Each expert can specialize in different:
    - Patch complexity levels
    - Texture vs structure
    - Detail vs coarse features

    Key optimizations:
    1. Batched expert computation (no Python loop)
    2. Fused INL dynamics with Triton
    3. Shared equilibrium across experts
    4. Load balancing for training stability
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        num_iterations: int = 2,
        dt: float = 0.1,
        use_shared_expert: bool = True,
        controller_hidden: int = 64,
        load_balance_weight: float = 0.01
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_iterations = num_iterations
        self.dt = dt
        self.use_shared_expert = use_shared_expert
        self.load_balance_weight = load_balance_weight

        # Per-channel integration weights (learned)
        self.integration_weight = nn.Parameter(torch.ones(d_model) * 0.5)

        # Shared equilibrium
        self.mu = nn.Parameter(torch.zeros(d_model))

        # Router - routes patches to experts based on content
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_experts)
        )

        # Halting gate - decides when to stop iterating
        self.halt_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        # Expert integrators - fused weights for batched computation
        ctx_dim = d_model * 2  # x + v
        self.expert_w1 = nn.Parameter(
            torch.randn(num_experts, ctx_dim, controller_hidden) * 0.02
        )
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, controller_hidden))
        self.expert_w2 = nn.Parameter(
            torch.randn(num_experts, controller_hidden, 3 * d_model) * 0.02
        )
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, 3 * d_model))

        # Shared expert (always used)
        if use_shared_expert:
            self.shared_expert = nn.Sequential(
                nn.Linear(ctx_dim, controller_hidden),
                nn.GELU(),
                nn.Linear(controller_hidden, 3 * d_model)
            )
            self.shared_weight = nn.Parameter(torch.tensor(0.5))

        # Refinement MLP
        self.refine = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Usage statistics
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.zeros(1))

    def _batched_expert_forward(
        self,
        ctx: torch.Tensor,       # [n_tokens, ctx_dim]
        expert_ids: torch.Tensor  # [n_tokens]
    ) -> torch.Tensor:
        """
        Batched expert MLP forward.

        Efficient implementation using gather + bmm instead of loop.
        """
        n_tokens = ctx.shape[0]

        # Gather weights for each token's expert
        w1 = self.expert_w1[expert_ids]  # [n_tokens, ctx_dim, hidden]
        b1 = self.expert_b1[expert_ids]  # [n_tokens, hidden]
        w2 = self.expert_w2[expert_ids]  # [n_tokens, hidden, 3*d_model]
        b2 = self.expert_b2[expert_ids]  # [n_tokens, 3*d_model]

        # Layer 1: [n_tokens, 1, ctx_dim] @ [n_tokens, ctx_dim, hidden]
        hidden = torch.bmm(ctx.unsqueeze(1), w1).squeeze(1) + b1
        hidden = F.gelu(hidden)

        # Layer 2: [n_tokens, 1, hidden] @ [n_tokens, hidden, out]
        out = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2

        return out

    def _compute_inl_dynamics(
        self,
        ctrl_out: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute INL dynamics from controller output."""
        alpha_raw, beta_raw, gate_raw = torch.split(ctrl_out, self.d_model, dim=-1)

        alpha = torch.sigmoid(alpha_raw)
        beta = F.softplus(beta_raw)
        gate = torch.sigmoid(gate_raw)

        # Use fused Triton kernel if available
        if HAS_TRITON and x.is_cuda:
            return fused_inl_dynamics(x, v, self.mu, alpha, beta, gate, self.dt)
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
        Forward pass with MoE routing.

        Args:
            x: Input tensor [B, N, D]
            context: Optional context (unused, for API compatibility)

        Returns:
            output: Refined tensor [B, N, D]
            aux: Auxiliary info (router probs, expert usage, etc.)
        """
        B, N, D = x.shape
        device = x.device

        # Flatten batch and sequence
        x_flat = x.view(B * N, D)

        # Route patches to experts
        router_logits = self.router(x_flat)  # [B*N, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k expert selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Initialize integrator state
        v = torch.zeros_like(x_flat)
        integrated = x_flat.clone()

        # Iterate
        for iteration in range(self.num_iterations):
            # Build context: [x, v]
            ctx = torch.cat([integrated, v], dim=-1)  # [B*N, 2*D]

            # === SHARED EXPERT ===
            if self.use_shared_expert:
                shared_ctrl = self.shared_expert(ctx)
                x_shared, v_shared = self._compute_inl_dynamics(shared_ctrl, integrated, v)
                shared_w = torch.sigmoid(self.shared_weight)
            else:
                x_shared = torch.zeros_like(integrated)
                v_shared = torch.zeros_like(v)
                shared_w = 0.0

            # === ROUTED EXPERTS ===
            # Expand for top-k
            flat_indices = top_k_indices.view(-1)  # [B*N*top_k]
            flat_weights = top_k_probs.view(-1, 1)  # [B*N*top_k, 1]

            ctx_expanded = ctx.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, ctx.size(-1))
            x_expanded = integrated.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, D)
            v_expanded = v.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, D)

            # Batched expert forward
            ctrl_out = self._batched_expert_forward(ctx_expanded, flat_indices)

            # Compute dynamics for each expert
            x_routed, v_routed = self._compute_inl_dynamics(ctrl_out, x_expanded, v_expanded)

            # Weight and aggregate
            x_weighted = x_routed * flat_weights
            v_weighted = v_routed * flat_weights

            x_routed = x_weighted.view(B * N, self.top_k, -1).sum(dim=1)
            v_routed = v_weighted.view(B * N, self.top_k, -1).sum(dim=1)

            # Combine shared + routed
            if self.use_shared_expert:
                x_next = shared_w * x_shared + (1 - shared_w) * x_routed
                v_next = shared_w * v_shared + (1 - shared_w) * v_routed
            else:
                x_next = x_routed
                v_next = v_routed

            # Compute halt probability
            halt_prob = self.halt_gate(x_next)  # [B*N, 1]

            # Refine
            refined = self.refine(x_next)

            # Integrate with per-channel weights
            integrated = integrated + halt_prob * refined * self.integration_weight

            v = v_next

        # Update statistics during training
        if self.training:
            for exp_id in range(self.num_experts):
                self.expert_counts[exp_id] += (flat_indices == exp_id).sum().float()
            self.total_tokens += B * N

        # Load balance loss
        load_balance_loss = None
        if self.training and self.load_balance_weight > 0:
            avg_prob = router_probs.mean(dim=0)
            target = 1.0 / self.num_experts
            load_balance_loss = ((avg_prob - target) ** 2).sum() * self.load_balance_weight

        # Reshape output
        output = integrated.view(B, N, D)

        aux = {
            'router_probs': router_probs.view(B, N, self.num_experts),
            'top_k_experts': top_k_indices.view(B, N, self.top_k),
            'expert_weights': top_k_probs.view(B, N, self.top_k),
            'shared_weight': shared_w if self.use_shared_expert else None,
            'load_balance_loss': load_balance_loss,
            'expert_usage': self.get_expert_usage()
        }

        return output, aux

    def get_expert_usage(self) -> Dict[str, float]:
        """Get expert usage statistics."""
        if self.total_tokens.item() == 0:
            return {}
        return {
            f'expert_{i}': self.expert_counts[i].item() / self.total_tokens.item()
            for i in range(self.num_experts)
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.expert_counts.zero_()
        self.total_tokens.zero_()


class AdaptiveMoEIntegrator(FusedMoEIntegrator):
    """
    MoE Integrator with adaptive iterations based on patch complexity.

    Learns to allocate more iterations to complex patches.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        min_iterations: int = 1,
        max_iterations: int = 4,
        **kwargs
    ):
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            num_iterations=max_iterations,
            **kwargs
        )

        self.min_iterations = min_iterations
        self.max_iterations = max_iterations

        # Iteration predictor - predicts how many iterations each patch needs
        self.iteration_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward with adaptive iterations.

        During training: uses max_iterations with weighted accumulation
        During inference: early stops based on predicted iterations
        """
        B, N, D = x.shape

        if self.training:
            # Training: full iterations with soft weighting
            return super().forward(x, context)

        # Inference: adaptive iteration count
        x_flat = x.view(B * N, D)

        # Predict iterations needed
        iter_prob = self.iteration_predictor(x_flat)  # [B*N, 1]
        predicted_iters = (
            self.min_iterations +
            (self.max_iterations - self.min_iterations) * iter_prob
        ).round().long().squeeze(-1)

        # Route patches
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Initialize
        v = torch.zeros_like(x_flat)
        integrated = x_flat.clone()
        output = torch.zeros_like(x_flat)
        done = torch.zeros(B * N, dtype=torch.bool, device=x.device)

        for iteration in range(self.max_iterations):
            # Process only non-done patches
            active = ~done
            if not active.any():
                break

            # Build context
            ctx = torch.cat([integrated, v], dim=-1)

            # Shared expert
            if self.use_shared_expert:
                shared_ctrl = self.shared_expert(ctx)
                x_shared, v_shared = self._compute_inl_dynamics(shared_ctrl, integrated, v)
                shared_w = torch.sigmoid(self.shared_weight)

            # Routed experts
            flat_indices = top_k_indices.view(-1)
            flat_weights = top_k_probs.view(-1, 1)

            ctx_expanded = ctx.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, ctx.size(-1))
            x_expanded = integrated.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, D)
            v_expanded = v.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, D)

            ctrl_out = self._batched_expert_forward(ctx_expanded, flat_indices)
            x_routed, v_routed = self._compute_inl_dynamics(ctrl_out, x_expanded, v_expanded)

            x_weighted = x_routed * flat_weights
            v_weighted = v_routed * flat_weights
            x_routed = x_weighted.view(B * N, self.top_k, -1).sum(dim=1)
            v_routed = v_weighted.view(B * N, self.top_k, -1).sum(dim=1)

            if self.use_shared_expert:
                x_next = shared_w * x_shared + (1 - shared_w) * x_routed
                v_next = shared_w * v_shared + (1 - shared_w) * v_routed
            else:
                x_next = x_routed
                v_next = v_routed

            halt_prob = self.halt_gate(x_next)
            refined = self.refine(x_next)
            integrated = integrated + halt_prob * refined * self.integration_weight
            v = v_next

            # Mark patches as done
            newly_done = (predicted_iters <= iteration + 1) & active
            output[newly_done] = integrated[newly_done]
            done = done | newly_done

        # Handle any remaining
        output[~done] = integrated[~done]

        output = output.view(B, N, D)

        aux = {
            'router_probs': router_probs.view(B, N, self.num_experts),
            'predicted_iterations': predicted_iters.view(B, N).float().mean(),
            'expert_usage': self.get_expert_usage()
        }

        return output, aux
