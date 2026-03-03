"""
Geodesic State Module (GSM) — the core computational unit of LF-SSM.

Implements §4.3 / Figure 3 of the paper.

Data-flow
=========
Input X ∈ R^{T×D}
  → Linear D→E + 1-D DW-Conv + SiLU  (shared expansion)
  → Branch 1 – **State evolution branch**
        Linear_B : E→N  (input projection)
        Linear_Δ : E→1  (step size)
        Geodesic state evolution  (Algorithm 1)
        C : N→E  (output projection)
  → Branch 2 – **Gating branch**
        Linear_G : E→E  (Eq. 10)
        SiLU activation
  → G ⊙ (C H)  element-wise  (Eq. 11)
  → Linear_O : E→D  (output projection)
"""

import torch
import torch.nn as nn
from torch import Tensor

from lf_ssm.geodesic_ops import (
    geodesic_state_evolution,
    normalize_to_sphere,
)


class GeodesicStateModule(nn.Module):
    """Geodesic State Module (GSM).

    Parameters
    ----------
    embed_dim : int
        Input / output token dimension D.
    state_dim : int
        Geodesic state dimension N  (state lives on S^{N-1}).
    expand_ratio : int
        E = D × expand_ratio.
    conv_kernel_size : int
        Kernel size for the depth-wise 1-D convolution.
    alpha : float
        Confidence weight for prior velocity momentum.
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        state_dim: int = 64,
        expand_ratio: int = 2,
        conv_kernel_size: int = 3,
        alpha: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.expanded_dim = embed_dim * expand_ratio
        self.alpha = alpha
        self.eps = eps

        # ── Shared expansion ───────────────────────────────────────
        self.in_proj = nn.Linear(embed_dim, self.expanded_dim)
        self.conv1d = nn.Conv1d(
            self.expanded_dim,
            self.expanded_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=self.expanded_dim,          # depth-wise
        )
        self.act = nn.SiLU()

        # ── State evolution branch ─────────────────────────────────
        # Linear_B : E → N   (input projection, Eq. 9)
        self.B_proj = nn.Linear(self.expanded_dim, state_dim)
        # Linear_Δ : E → 1   (step size, Eq. 9)
        self.delta_proj = nn.Linear(self.expanded_dim, 1)
        # C : N → E          (output projection, Eq. 11)
        self.C_proj = nn.Linear(state_dim, self.expanded_dim)
        # Learnable initial state ĥ₀ (normalised in forward)
        self.h0_param = nn.Parameter(torch.randn(state_dim))

        # ── Gating branch (Eq. 10) ─────────────────────────────────
        self.gate_proj = nn.Linear(self.expanded_dim, self.expanded_dim)
        self.gate_act = nn.SiLU()

        # ── Output projection (Eq. 11) ─────────────────────────────
        self.out_proj = nn.Linear(self.expanded_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
            Input token sequence.

        Returns
        -------
        Tensor, shape (B, T, D)
            Output token sequence.
        """
        B, T, D = x.shape

        # ── Shared expansion + conv + activation ───────────────────
        x_exp = self.in_proj(x)                      # (B, T, E)
        # Conv1d expects (B, C, T)
        x_conv = self.conv1d(x_exp.transpose(1, 2)).transpose(1, 2)  # (B, T, E)
        x_tilde = self.act(x_conv)                   # (B, T, E)

        # ── State evolution branch ─────────────────────────────────
        h0 = normalize_to_sphere(self.h0_param, self.eps)  # (N,)
        H = geodesic_state_evolution(
            x_seq=x_tilde,
            B_proj=self.B_proj,
            delta_proj=self.delta_proj,
            h0=h0.unsqueeze(0),                      # (1, N)
            alpha=self.alpha,
            eps=self.eps,
        )                                            # (B, T, N)
        state_out = self.C_proj(H)                   # (B, T, E)

        # ── Gating branch (Eq. 10) ─────────────────────────────────
        gate = self.gate_act(self.gate_proj(x_tilde))  # (B, T, E)

        # ── Combine: G ⊙ (CH)  then project (Eq. 11) ──────────────
        y = self.out_proj(gate * state_out)           # (B, T, D)
        return y
