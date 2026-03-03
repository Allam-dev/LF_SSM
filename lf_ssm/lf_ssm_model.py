"""
Full LF-SSM tracker model.

Implements the overall architecture from §4.1 / Figure 2 / Algorithms 2–3:

    1. Patch embedding (template + search)
    2. Concatenated token sequence
    3. Stacked GSM Blocks
    4. Extract search tokens → spatial grid
    5. Prediction head → (cls_score, bbox_pred)

Supports three modes:
    - **Training** : `forward(template, search)` → (cls, bbox)
    - **Template caching** : `set_template(template)` encodes once
    - **Inference** : `track(search)` uses cached template → (cls, bbox)
"""

import torch
import torch.nn as nn
from torch import Tensor

from lf_ssm.config import ModelConfig
from lf_ssm.patch_embed import PatchEmbed
from lf_ssm.gsm_block import GSMBlock
from lf_ssm.prediction_head import PredictionHead


class LFSSM(nn.Module):
    """LF-SSM: Lightweight HiPPO-Free State Space Model for UAV Tracking.

    Parameters
    ----------
    cfg : ModelConfig
        Full model configuration.  Use the factory helpers
        ``LF_SSM_S()``, ``LF_SSM_M()``, ``LF_SSM_L()`` for the paper
        variants, or build a custom ``ModelConfig``.

    Example
    -------
    >>> from lf_ssm import LFSSM, LF_SSM_S
    >>> model = LFSSM(LF_SSM_S())
    >>> template = torch.randn(1, 3, 128, 128)
    >>> search   = torch.randn(1, 3, 256, 256)
    >>> cls, bbox = model(template, search)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ── Patch embeddings (§4.1) ────────────────────────────────
        self.template_embed = PatchEmbed(
            img_size=cfg.img_size_template,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )
        self.search_embed = PatchEmbed(
            img_size=cfg.img_size_search,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )

        # ── Stacked GSM Blocks (§4.4) ─────────────────────────────
        # Linearly increasing drop-path rate
        dpr = [
            cfg.drop_path_rate * i / max(cfg.num_blocks - 1, 1)
            for i in range(cfg.num_blocks)
        ]
        self.blocks = nn.ModuleList([
            GSMBlock(
                embed_dim=cfg.embed_dim,
                state_dim=cfg.state_dim,
                expand_ratio=cfg.expand_ratio,
                conv_kernel_size=cfg.conv_kernel_size,
                alpha=cfg.alpha,
                eps=cfg.eps,
                drop_path=dpr[i],
            )
            for i in range(cfg.num_blocks)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        # ── Prediction head (§4.5) ─────────────────────────────────
        self.head = PredictionHead(
            embed_dim=cfg.embed_dim,
            num_classes=cfg.num_classes,
        )

        # ── Template cache for inference ───────────────────────────
        self._cached_template_tokens: Tensor | None = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ── Forward (training) ─────────────────────────────────────────

    def forward(self, template: Tensor, search: Tensor) -> tuple[Tensor, Tensor]:
        """Training forward pass (Algorithm 2).

        Parameters
        ----------
        template : Tensor, shape (B, C, H_t, W_t)
        search   : Tensor, shape (B, C, H_s, W_s)

        Returns
        -------
        cls_score : Tensor, shape (B, num_classes, H_s/P, W_s/P)
        bbox_pred : Tensor, shape (B, 4, H_s/P, W_s/P)
        """
        z_t = self.template_embed(template)       # (B, Tt, D)
        z_s = self.search_embed(search)            # (B, Ts, D)
        z = torch.cat([z_t, z_s], dim=1)          # (B, Tt+Ts, D)

        # Stacked GSM Blocks
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)

        # Extract search tokens and reshape to spatial grid
        z_search = z[:, self.cfg.num_patches_template:, :]   # (B, Ts, D)
        feat_size = self.cfg.search_feat_size                # H_s/P
        z_spatial = z_search.transpose(1, 2).reshape(
            -1, self.cfg.embed_dim, feat_size, feat_size
        )                                                    # (B, D, f, f)

        cls_score, bbox_pred = self.head(z_spatial)
        return cls_score, bbox_pred

    # ── Inference helpers (Algorithm 3) ────────────────────────────

    @torch.no_grad()
    def set_template(self, template: Tensor) -> None:
        """Encode template once and cache (Algorithm 3, line 1).

        Parameters
        ----------
        template : Tensor, shape (B, C, H_t, W_t)
        """
        self._cached_template_tokens = self.template_embed(template)

    @torch.no_grad()
    def track(self, search: Tensor) -> tuple[Tensor, Tensor]:
        """Track a single search frame using cached template (Algorithm 3).

        Call ``set_template`` first.

        Parameters
        ----------
        search : Tensor, shape (B, C, H_s, W_s)

        Returns
        -------
        cls_score, bbox_pred
        """
        assert self._cached_template_tokens is not None, (
            "Call set_template() before track()"
        )
        z_s = self.search_embed(search)
        z = torch.cat([self._cached_template_tokens, z_s], dim=1)

        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)

        z_search = z[:, self.cfg.num_patches_template:, :]
        feat_size = self.cfg.search_feat_size
        z_spatial = z_search.transpose(1, 2).reshape(
            -1, self.cfg.embed_dim, feat_size, feat_size
        )

        cls_score, bbox_pred = self.head(z_spatial)
        return cls_score, bbox_pred

    def get_param_count(self) -> dict[str, int]:
        """Return parameter counts by component."""
        counts = {}
        counts["template_embed"] = sum(p.numel() for p in self.template_embed.parameters())
        counts["search_embed"] = sum(p.numel() for p in self.search_embed.parameters())
        counts["blocks"] = sum(p.numel() for p in self.blocks.parameters())
        counts["head"] = sum(p.numel() for p in self.head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
