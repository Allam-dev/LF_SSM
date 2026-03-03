"""
Prediction head for LF-SSM (§4.5).

Takes the enhanced search-region features (reshaped to a spatial grid) and
produces:
  - **Classification map** – foreground / background score per spatial cell
    (trained with Focal Loss).
  - **Regression map** – normalised bounding-box offsets (x, y, w, h)
    per spatial cell (trained with GIoU + L1).
"""

import torch
import torch.nn as nn
from torch import Tensor


class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU  helper."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class PredictionHead(nn.Module):
    """Classification + Regression prediction head.

    Parameters
    ----------
    embed_dim : int
        Input channel count (= token dimension D).
    num_classes : int
        Number of foreground classes (typically 1).
    hidden_dim : int or None
        Hidden channels inside the conv stacks. Defaults to embed_dim.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 1,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        # ── Classification branch ──────────────────────────────────
        self.cls_head = nn.Sequential(
            ConvBNReLU(embed_dim, hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

        # ── Regression branch ──────────────────────────────────────
        self.reg_head = nn.Sequential(
            ConvBNReLU(embed_dim, hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim),
            ConvBNReLU(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 4, kernel_size=1),  # (x, y, w, h)
            nn.Sigmoid(),  # normalised to [0,1]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, D, H_s, W_s)
            Search-region features reshaped to a spatial grid.

        Returns
        -------
        cls_score : Tensor, shape (B, num_classes, H_s, W_s)
        bbox_pred : Tensor, shape (B, 4, H_s, W_s)
            Normalised (x, y, w, h).
        """
        cls_score = self.cls_head(x)
        bbox_pred = self.reg_head(x)
        return cls_score, bbox_pred
