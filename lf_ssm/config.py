"""
Model configuration for LF-SSM.

All tunable parameters are documented with their paper references.
Use the pre-defined LF_SSM_S / LF_SSM_M / LF_SSM_L configs or create
custom variants by modifying the ModelConfig fields.

How to change parameter counts
==============================
The total parameter count is primarily controlled by three knobs:

1. **num_blocks (L)** – Number of stacked GSM Blocks.
   More blocks ⇒ more parameters and deeper feature interaction.
   Paper variants: S=6, M=12, L=18.

2. **embed_dim (D)** – Token / feature dimension.
   Affects every linear layer. Doubling D roughly quadruples params.

3. **state_dim (N)** – Dimension of the geodesic state on S^{N-1}.
   Controls the capacity of the manifold representation.
   Paper recommends N=64 (Table 8). N=16–256 tested.

4. **expand_ratio** – Multiplier for the expanded dimension E = D * expand_ratio.
   Higher ratio ⇒ wider GSM internal processing.

Secondary knobs: patch_size (changes token count, not param count much),
conv_kernel_size, alpha (prior velocity weight).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Complete configuration for the LF-SSM model.

    Attributes
    ----------
    img_size_search : int
        Spatial size of the search region crop (§6.1.3). Default 256.
    img_size_template : int
        Spatial size of the template region crop (§6.1.3). Default 128.
    patch_size : int
        Side length of each non-overlapping patch (§4.1). Default 16.
    in_channels : int
        Number of input image channels. Default 3 (RGB).
    embed_dim : int
        Token / feature dimension D (§4.3). Default 256.
    state_dim : int
        Geodesic state dimension N on S^{N-1} (§4.2, §6.1.3, Table 8). Default 64.
    expand_ratio : int
        Expansion ratio so E = embed_dim * expand_ratio (§4.3). Default 2.
    num_blocks : int
        Number of stacked GSM Blocks L (§4.4, §6.1.3). Default 18 (LF-SSM-L).
    conv_kernel_size : int
        Kernel size for the 1-D depthwise convolution inside GSM (§4.3). Default 3.
    alpha : float
        Confidence weight for prior velocity momentum (Algorithm 1). Default 0.5.
    eps : float
        Numerical stability constant ε (§4.2). Default 1e-6.
    num_classes : int
        Number of foreground classes for the classification head. Default 1.
    lambda_iou : float
        Weight for GIoU loss (Eq. 13). Default 2.0.
    lambda_l1 : float
        Weight for L1 loss (Eq. 13). Default 5.0.
    drop_path_rate : float
        Stochastic depth rate spread across blocks. Default 0.1.
    """

    # ── Image & patch ──────────────────────────────────────────────
    img_size_search: int = 256
    img_size_template: int = 128
    patch_size: int = 16
    in_channels: int = 3

    # ── Dimensions ─────────────────────────────────────────────────
    embed_dim: int = 256          # D
    state_dim: int = 64           # N  (state lives on S^{N-1})
    expand_ratio: int = 2         # E = embed_dim * expand_ratio

    # ── Architecture ───────────────────────────────────────────────
    num_blocks: int = 18          # L  (S=6, M=12, L=18)
    conv_kernel_size: int = 3

    # ── Geodesic hyper-parameters ──────────────────────────────────
    alpha: float = 0.5            # Prior velocity confidence
    eps: float = 1e-6             # Numerical stability

    # ── Head ───────────────────────────────────────────────────────
    num_classes: int = 1

    # ── Loss weights (Eq. 13) ──────────────────────────────────────
    lambda_iou: float = 2.0
    lambda_l1: float = 5.0

    # ── Regularization ─────────────────────────────────────────────
    drop_path_rate: float = 0.1

    # ── Derived (computed at post-init) ────────────────────────────
    @property
    def expanded_dim(self) -> int:
        """E = D × expand_ratio  (§4.3)."""
        return self.embed_dim * self.expand_ratio

    @property
    def num_patches_search(self) -> int:
        """Number of search tokens Ts = (img_size_search / patch_size)²."""
        return (self.img_size_search // self.patch_size) ** 2

    @property
    def num_patches_template(self) -> int:
        """Number of template tokens Tt = (img_size_template / patch_size)²."""
        return (self.img_size_template // self.patch_size) ** 2

    @property
    def num_patches_total(self) -> int:
        """T = Tt + Ts  (total token count after concatenation)."""
        return self.num_patches_template + self.num_patches_search

    @property
    def search_feat_size(self) -> int:
        """Spatial side of the search feature map."""
        return self.img_size_search // self.patch_size

    @property
    def template_feat_size(self) -> int:
        """Spatial side of the template feature map."""
        return self.img_size_template // self.patch_size


# ── Pre-defined variants (§6.1.3) ──────────────────────────────────

def LF_SSM_S() -> ModelConfig:
    """LF-SSM-S: 6 GSM Blocks, ~18.5 M params."""
    return ModelConfig(num_blocks=6)


def LF_SSM_M() -> ModelConfig:
    """LF-SSM-M: 12 GSM Blocks, ~35.2 M params."""
    return ModelConfig(num_blocks=12)


def LF_SSM_L() -> ModelConfig:
    """LF-SSM-L: 18 GSM Blocks, ~52.8 M params."""
    return ModelConfig(num_blocks=18)


def LF_SSM_Nano() -> ModelConfig:
    """LF-SSM-Nano: ~5 M params.  Tuned for resource-constrained training.

    Configuration rationale
    -----------------------
    - D=176 : enough representation capacity (quadratic impact on params)
    - N=24  : S^23 manifold still has 23 degrees of freedom
    - L=5   : 5 GSM Blocks for feature refinement
    - E=352 : expand_ratio=2 keeps the GSM internal capacity ratio
    - drop_path=0.05 : lighter regularisation for smaller model
    """
    return ModelConfig(
        embed_dim=176,
        state_dim=24,
        num_blocks=5,
        expand_ratio=2,
        drop_path_rate=0.05,
    )
