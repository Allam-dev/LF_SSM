"""
GSM Block — the repeated building block of the LF-SSM backbone.

Implements §4.4 / Eq. 12:
    Z' = LayerNorm(Z)
    Z_out = Z + Linear([GSM_fwd(Z') ; GSM_bwd(Z')])

Bidirectional processing: the forward GSM scans tokens left→right,
the backward GSM scans tokens right→left.  Outputs are concatenated
along the feature dimension (2D) and projected back to D.
"""

import torch
import torch.nn as nn
from torch import Tensor

from lf_ssm.gsm_module import GeodesicStateModule


class DropPath(nn.Module):
    """Stochastic depth (drop path) for regularisation."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class GSMBlock(nn.Module):
    """A single GSM Block with bidirectional geodesic state evolution.

    Parameters
    ----------
    embed_dim : int
        Token dimension D.
    state_dim : int
        Geodesic state dimension N.
    expand_ratio : int
        Expansion ratio for E = D × expand_ratio.
    conv_kernel_size : int
        Kernel size for 1-D depthwise conv inside GSM.
    alpha : float
        Prior velocity confidence weight.
    eps : float
        Numerical stability constant.
    drop_path : float
        Stochastic depth drop probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        state_dim: int = 64,
        expand_ratio: int = 2,
        conv_kernel_size: int = 3,
        alpha: float = 0.5,
        eps: float = 1e-6,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)

        # Forward and backward GSM (§4.4)
        gsm_kwargs = dict(
            embed_dim=embed_dim,
            state_dim=state_dim,
            expand_ratio=expand_ratio,
            conv_kernel_size=conv_kernel_size,
            alpha=alpha,
            eps=eps,
        )
        self.gsm_fwd = GeodesicStateModule(**gsm_kwargs)
        self.gsm_bwd = GeodesicStateModule(**gsm_kwargs)

        # Linear projection: 2D → D  (concat of fwd + bwd outputs, Eq. 12)
        self.merge_proj = nn.Linear(2 * embed_dim, embed_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor, shape (B, T, D)
            Input token sequence.

        Returns
        -------
        Tensor, shape (B, T, D)
            Output with residual connection.
        """
        # ── LayerNorm ──────────────────────────────────────────────
        z_norm = self.norm(z)                        # (B, T, D)

        # ── Bidirectional GSM ──────────────────────────────────────
        out_fwd = self.gsm_fwd(z_norm)               # (B, T, D)
        # Reverse token order → process → reverse back
        z_rev = z_norm.flip(dims=[1])                # (B, T, D)
        out_bwd = self.gsm_bwd(z_rev).flip(dims=[1]) # (B, T, D)

        # ── Concatenate + project  (Eq. 12) ────────────────────────
        merged = torch.cat([out_fwd, out_bwd], dim=-1)  # (B, T, 2D)
        merged = self.merge_proj(merged)             # (B, T, D)

        # ── Residual connection ────────────────────────────────────
        return z + self.drop_path(merged)
