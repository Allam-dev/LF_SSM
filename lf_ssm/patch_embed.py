"""
Patch embedding layer for LF-SSM.

Converts raw images into flat sequences of D-dimensional patch tokens
with learnable positional embeddings.

Template : 128×128 → (128/16)² = 64 tokens  (default)
Search   : 256×256 → (256/16)² = 256 tokens (default)
"""

import torch
import torch.nn as nn
from torch import Tensor


class PatchEmbed(nn.Module):
    """Image-to-token patch embedding with positional encoding.

    Parameters
    ----------
    img_size : int
        Spatial size of the input image (assumed square).
    patch_size : int
        Side length of each non-overlapping patch.
    in_channels : int
        Number of input channels (3 for RGB).
    embed_dim : int
        Output token dimension D.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # Xavier-uniform init for position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C, H, W)

        Returns
        -------
        Tensor, shape (B, num_patches, D)
        """
        x = self.proj(x)            # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        x = self.norm(x)
        x = x + self.pos_embed
        return x
