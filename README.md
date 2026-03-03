# LF-SSM: Lightweight HiPPO-Free State Space Model

PyTorch implementation of **"LF-SSM: Lightweight HiPPO-Free State Space Model for Real-Time UAV Tracking"** (Wang et al., *Drones* 2026, 10, 102).

## Key Idea

LF-SSM replaces the HiPPO-based state transitions (fixed Legendre polynomial bases) used in Mamba/S4 with **geodesic dynamics on the unit sphere S^{N-1}**. This:
- **Eliminates** complex discretisation and specialised CUDA kernels
- **Provides adaptive** local coordinate systems via tangent spaces
- **Preserves** geometric structure of tracking features
- Runs at **69 FPS on Jetson Orin Nano** with only 18.5M parameters

## Architecture Overview

```
Template (128×128) ─→ PatchEmbed ─→ Tt tokens ─┐
                                                 ├─→ [Zt; Zs] ─→ L × GSM Blocks ─→ Search tokens ─→ Pred Head ─→ (cls, bbox)
Search  (256×256) ─→ PatchEmbed ─→ Ts tokens ─┘
```

Each **GSM Block** (§4.4):
```
Z_out = Z + Linear([GSM_fwd(LayerNorm(Z)) ; GSM_bwd(LayerNorm(Z))])
```

Each **GSM** (§4.3): expansion → 1D Conv → two branches (geodesic state evolution + gating) → merge.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from lf_ssm import LFSSM, LF_SSM_S, LF_SSM_M, LF_SSM_L

# ── Create a model variant ─────────────────────────
model = LFSSM(LF_SSM_S())   # 6 blocks, ~18.5M params
# model = LFSSM(LF_SSM_M())  # 12 blocks, ~35.2M params
# model = LFSSM(LF_SSM_L())  # 18 blocks, ~52.8M params

# ── Training forward pass ──────────────────────────
template = torch.randn(1, 3, 128, 128)
search   = torch.randn(1, 3, 256, 256)
cls_score, bbox_pred = model(template, search)
# cls_score: (1, 1, 16, 16)   bbox_pred: (1, 4, 16, 16)

# ── Inference with template caching ────────────────
model.eval()
model.set_template(template)         # encode once
cls, bbox = model.track(search)      # fast per-frame tracking
```

## How to Change Parameters

All parameters live in `ModelConfig` (see `lf_ssm/config.py`).

### Main parameter knobs

| Parameter | Symbol | Default | Effect |
|---|---|---|---|
| `num_blocks` | L | 18 | **Biggest impact on param count.** S=6, M=12, L=18 |
| `embed_dim` | D | 256 | Token dimension. Doubling D roughly 4× params |
| `state_dim` | N | 64 | State sphere dimension. Paper tests 16–256 (Table 8) |
| `expand_ratio` | E/D | 2 | Width of GSM internals. Higher = wider but more params |
| `patch_size` | P | 16 | Patch granularity. Smaller = more tokens (slower) |

### Creating a custom variant

```python
from lf_ssm import ModelConfig, LFSSM

cfg = ModelConfig(
    num_blocks=8,        # custom depth
    embed_dim=192,       # narrower
    state_dim=32,        # smaller manifold
    expand_ratio=2,
    img_size_search=192, # smaller search crop
    img_size_template=96,
    patch_size=16,
    alpha=0.3,           # less prior momentum
)
model = LFSSM(cfg)
print(model.get_param_count())
```

### Key hyperparameters

- **`alpha`** (default 0.5): Prior velocity confidence. Higher = stronger temporal momentum.
- **`eps`** (default 1e-6): Numerical stability for tangent-space operations.
- **`lambda_iou`** / **`lambda_l1`** (defaults 2.0 / 5.0): Loss balance weights (Eq. 13).
- **`drop_path_rate`** (default 0.1): Stochastic depth for regularisation.

## Project Structure

```
lf_ssm/
├── config.py          # ModelConfig with all tunable parameters
├── geodesic_ops.py    # Core math: tangent projection, exp map, Algorithm 1
├── gsm_module.py      # Geodesic State Module (Figure 3)
├── gsm_block.py       # GSM Block with bidirectional processing (Eq. 12)
├── patch_embed.py     # ViT-style patch embedding
├── prediction_head.py # Classification + regression head (§4.5)
├── losses.py          # Focal, GIoU, L1 losses (Eq. 13)
└── lf_ssm_model.py    # Full LF-SSM tracker (Algorithms 2–3)
```

## Running Tests

```bash
python -m pytest tests/test_lf_ssm.py -v
```

## Paper Reference

```
@article{wang2026lfssm,
  title={LF-SSM: Lightweight HiPPO-Free State Space Model for Real-Time UAV Tracking},
  author={Wang, Tianyu and Xu, Xinghua and Qiu, Shaohua and Sheng, Changchong
          and Wang, Di and Tian, Hui and Yu, Jiawei},
  journal={Drones},
  volume={10},
  number={2},
  pages={102},
  year={2026},
  publisher={MDPI}
}
```
