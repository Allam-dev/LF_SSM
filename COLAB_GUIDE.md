# Running LF-SSM on Google Colab with UAV Datasets

A complete, step-by-step guide to training and evaluating the LF-SSM tracker on Google Colab, including instructions for a **~5M parameter** configuration.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Upload the LF-SSM Code](#2-upload-the-lf-ssm-code)
3. [Download UAV Datasets](#3-download-uav-datasets)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Configure the Model (~5M Params)](#5-configure-the-model-5m-params)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Inference on Video](#8-inference-on-video)
9. [Tips & Troubleshooting](#9-tips--troubleshooting)

---

## 1. Environment Setup

Open a new Colab notebook and run these cells:

```python
# Cell 1: Check GPU availability
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

```python
# Cell 2: Install dependencies (Colab has PyTorch pre-installed)
!pip install -q gdown  # for dataset downloads from Google Drive
```

---

## 2. Upload the LF-SSM Code

### Option A: Upload from local machine

```python
# Cell 3a: Upload as zip
from google.colab import files
uploaded = files.upload()  # Select lf_ssm.zip

!unzip lf_ssm.zip -d /content/LF_SSM
```

### Option B: Clone from a GitHub repo (if you pushed the code)

```python
# Cell 3b: Clone from git (replace with your repo URL)
!git clone https://github.com/YOUR_USERNAME/LF_SSM.git /content/LF_SSM
```

### Option C: Mount Google Drive

```python
# Cell 3c: Mount Drive (if code is in Drive)
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/LF_SSM /content/LF_SSM
```

```python
# Cell 4: Add to Python path
import sys
sys.path.insert(0, '/content/LF_SSM')

# Verify import
from lf_ssm import LFSSM, ModelConfig
print("LF-SSM imported successfully!")
```

---

## 3. Download UAV Datasets

The paper uses four benchmarks. Here we detail **UAV123** (primary) and **VisDrone** (secondary).

### UAV123 Dataset

```python
# Cell 5: Download UAV123
import os
os.makedirs('/content/datasets', exist_ok=True)

# UAV123 (~3.5 GB)
!wget -q --show-progress -O /content/datasets/UAV123.zip \
    "https://uav123.s3.eu-central-1.amazonaws.com/UAV123.zip"

!unzip -q /content/datasets/UAV123.zip -d /content/datasets/
print("UAV123 extracted!")
```

The UAV123 dataset structure after extraction:
```
/content/datasets/UAV123/
├── data_seq/UAV123/         # Video frames organised by sequence
│   ├── bike1/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   ├── bird1/
│   └── ...
└── anno/UAV123/             # Ground truth annotations
    ├── bike1.txt            # Each line: x,y,w,h
    ├── bird1.txt
    └── ...
```

### VisDrone Dataset (optional)

```python
# Cell 6: Download VisDrone SOT (Single Object Tracking)
# Training split (~1.7 GB)
!wget -q --show-progress -O /content/datasets/VisDrone_train.zip \
    "https://drive.google.com/uc?export=download&id=1fK3bLPsU-_DuE5r3N8KXebh_ciK-5g6W"

!unzip -q /content/datasets/VisDrone_train.zip -d /content/datasets/VisDrone/
```

---

## 4. Dataset Preparation

Create the dataset loader that generates template-search pairs.

```python
# Cell 7: UAV Tracking Dataset

import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class UAVTrackingDataset(Dataset):
    """
    Generates template-search pairs from UAV tracking sequences.

    Each sample:
      - template: crop around target in a reference frame (128×128)
      - search:   crop around target in a nearby frame (256×256)
      - bbox:     ground-truth bounding box in the search crop (cx, cy, w, h) normalised to [0,1]
      - cls_label: Gaussian heatmap for classification
    """

    def __init__(
        self,
        data_root: str,
        anno_root: str,
        search_size: int = 256,
        template_size: int = 128,
        search_factor: float = 4.0,
        template_factor: float = 2.0,
        max_gap: int = 100,
    ):
        """
        Parameters
        ----------
        data_root : str
            Path to sequence frames, e.g. '/content/datasets/UAV123/data_seq/UAV123'
        anno_root : str
            Path to annotations, e.g. '/content/datasets/UAV123/anno/UAV123'
        search_size : int
            Search region crop size. Default 256.
        template_size : int
            Template region crop size. Default 128.
        search_factor : float
            Scale factor around target for search crop. Default 4.0.
        template_factor : float
            Scale factor around target for template crop. Default 2.0.
        max_gap : int
            Maximum frame gap between template and search. Default 100.
        """
        self.search_size = search_size
        self.template_size = template_size
        self.search_factor = search_factor
        self.template_factor = template_factor
        self.max_gap = max_gap

        # Collect all sequences
        self.sequences = []
        anno_files = sorted(glob.glob(os.path.join(anno_root, '*.txt')))
        for anno_file in anno_files:
            seq_name = os.path.splitext(os.path.basename(anno_file))[0]
            seq_dir = os.path.join(data_root, seq_name)
            if not os.path.isdir(seq_dir):
                continue

            # Read annotations (x, y, w, h) - some use comma, some use tab/space
            with open(anno_file, 'r') as f:
                lines = f.readlines()
            bboxes = []
            for line in lines:
                line = line.strip()
                if not line:
                    bboxes.append([float('nan')] * 4)  # placeholder for empty lines
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 4:
                    vals = [float(p) for p in parts[:4]]
                    bboxes.append(vals)
                else:
                    bboxes.append([float('nan')] * 4)

            # Get sorted frame paths
            frames = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
            if len(frames) == 0:
                frames = sorted(glob.glob(os.path.join(seq_dir, '*.png')))

            if len(bboxes) > 0 and len(frames) > 0:
                n = min(len(frames), len(bboxes))
                bboxes_arr = np.array(bboxes[:n])  # (N, 4): x, y, w, h
                self.sequences.append({
                    'name': seq_name,
                    'frames': frames[:n],
                    'bboxes': bboxes_arr,
                })

        print(f"Loaded {len(self.sequences)} sequences")

        # Build index: (seq_idx, frame_idx) — only for frames with valid bboxes
        self.samples = []
        skipped = 0
        for si, seq in enumerate(self.sequences):
            bboxes = seq['bboxes']
            for fi in range(len(seq['frames'])):
                bbox = bboxes[fi]
                # Skip frames with NaN, zero-area, or negative bboxes
                if np.any(np.isnan(bbox)) or bbox[2] <= 0 or bbox[3] <= 0:
                    skipped += 1
                    continue
                self.samples.append((si, fi))
        print(f"Valid samples: {len(self.samples)} (skipped {skipped} invalid frames)")

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def _crop_and_resize(self, image, bbox, crop_size, factor):
        """Crop a region around bbox and resize to crop_size."""
        x, y, w, h = bbox
        # Guard against NaN or invalid bboxes — fall back to image center
        img_h, img_w = image.shape[:2]
        if np.any(np.isnan([x, y, w, h])) or w <= 0 or h <= 0:
            x, y = img_w * 0.25, img_h * 0.25
            w, h = img_w * 0.5, img_h * 0.5
        cx, cy = x + w / 2, y + h / 2
        s = max(w, h) * factor
        s = max(s, 1.0)  # ensure s is never zero
        # Crop region
        x1 = int(cx - s / 2)
        y1 = int(cy - s / 2)
        x2 = int(cx + s / 2)
        y2 = int(cy + s / 2)

        # Pad if out of bounds
        img_h, img_w = image.shape[:2]
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_w)
        pad_bottom = max(0, y2 - img_h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        crop = image[y1:y2, x1:x2]
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = np.pad(crop,
                          ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                          mode='constant', constant_values=0)

        # Resize
        crop_pil = Image.fromarray(crop)
        crop_pil = crop_pil.resize((crop_size, crop_size), Image.BILINEAR)

        # Compute bbox in crop coordinates (normalised)
        # Target center relative to crop
        cx_crop = (cx - (x1 - pad_left)) / (s if s > 0 else 1)
        cy_crop = (cy - (y1 - pad_top)) / (s if s > 0 else 1)
        w_crop = w / (s if s > 0 else 1)
        h_crop = h / (s if s > 0 else 1)

        bbox_norm = np.array([cx_crop, cy_crop, w_crop, h_crop], dtype=np.float32)
        bbox_norm = np.clip(bbox_norm, 0, 1)

        return crop_pil, bbox_norm

    def _make_cls_label(self, bbox_norm, feat_size):
        """Create a Gaussian heatmap for classification."""
        cx, cy = bbox_norm[0], bbox_norm[1]
        label = np.zeros((1, feat_size, feat_size), dtype=np.float32)
        # Place a Gaussian at the target center
        for i in range(feat_size):
            for j in range(feat_size):
                gi = (i + 0.5) / feat_size
                gj = (j + 0.5) / feat_size
                dist_sq = (gi - cy) ** 2 + (gj - cx) ** 2
                label[0, i, j] = np.exp(-dist_sq / (2 * 0.05 ** 2))
        return torch.from_numpy(label)

    def _is_valid_bbox(self, bbox):
        """Check if a bounding box is valid (no NaN, positive area)."""
        return (not np.any(np.isnan(bbox))) and bbox[2] > 0 and bbox[3] > 0

    def _find_valid_search_frame(self, seq, frame_idx):
        """Find a nearby frame with a valid bbox for use as search frame."""
        n_frames = len(seq['frames'])
        # Try random frames within max_gap, up to 20 attempts
        for _ in range(20):
            lo = max(0, frame_idx - self.max_gap)
            hi = min(n_frames - 1, frame_idx + self.max_gap)
            candidate = random.randint(lo, hi)
            if self._is_valid_bbox(seq['bboxes'][candidate]):
                return candidate
        # Fallback: use the template frame itself (guaranteed valid by sample index)
        return frame_idx

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.samples[idx]
        seq = self.sequences[seq_idx]

        # Template frame = current frame (guaranteed valid by sample index)
        template_frame_idx = frame_idx

        # Search frame = random nearby frame with valid bbox
        search_frame_idx = self._find_valid_search_frame(seq, frame_idx)

        # Load images
        template_img = np.array(Image.open(seq['frames'][template_frame_idx]).convert('RGB'))
        search_img = np.array(Image.open(seq['frames'][search_frame_idx]).convert('RGB'))

        template_bbox = seq['bboxes'][template_frame_idx]
        search_bbox = seq['bboxes'][search_frame_idx]

        # Crop and resize
        template_crop, _ = self._crop_and_resize(
            template_img, template_bbox, self.template_size, self.template_factor
        )
        search_crop, search_bbox_norm = self._crop_and_resize(
            search_img, search_bbox, self.search_size, self.search_factor
        )

        # Transform to tensors
        template_tensor = self.transform(template_crop)
        search_tensor = self.transform(search_crop)

        # Classification label (heatmap on the feature grid)
        feat_size = self.search_size // 16  # depends on patch_size
        cls_label = self._make_cls_label(search_bbox_norm, feat_size)

        return {
            'template': template_tensor,      # (3, 128, 128)
            'search': search_tensor,           # (3, 256, 256)
            'bbox': torch.from_numpy(search_bbox_norm),  # (4,) normalised
            'cls_label': cls_label,            # (1, feat_size, feat_size)
        }
```

```python
# Cell 8: Create DataLoader

dataset = UAVTrackingDataset(
    data_root='/content/datasets/UAV123/data_seq/UAV123',
    anno_root='/content/datasets/UAV123/anno/UAV123',
    search_size=256,
    template_size=128,
)

# Split into train/val (80/20)
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
```

---

## 5. Configure the Model (~5M Params)

The default LF-SSM-S has ~18.5M parameters. To get **~5M parameters**, reduce the architectural dimensions:

### Parameter knobs and their effects

| Parameter | Default (18.5M) | 5M config | Effect on params |
|---|---|---|---|
| `num_blocks` (L) | 6 | **5** | Fewer blocks = linear reduction |
| `embed_dim` (D) | 256 | **176** | Token dim — quadratic effect |
| `state_dim` (N) | 64 | **24** | Smaller manifold S^23 |
| `expand_ratio` | 2 | 2 | Keep same |

### Why these specific values?

- **D=176** keeps enough representation capacity (paper shows D matters most)
- **N=24** — The unit sphere S^23 still has 23 degrees of freedom, sufficient for capturing directional tracking features. Paper shows N=16 still gets 70.2% AUC (Table 8)
- **L=5** — 5 blocks still allows multiples layers of feature refinement
- **expand_ratio=2** — Keeping E=2D preserves the GSM internal capacity ratio

```python
# Cell 9: Create ~5M parameter model

# ⚠️ IMPORTANT: Make sure LF_SSM is in Python path
# (adjust the path if you placed the code in a different location)
import sys
sys.path.insert(0, '/content/LF_SSM')

from lf_ssm import ModelConfig, LFSSM

# ──────────────────────────────────────────────────────────────────
# ~5M PARAMETER CONFIGURATION
# ──────────────────────────────────────────────────────────────────
cfg_5m = ModelConfig(
    # Image sizes (keep default for standard UAV benchmarks)
    img_size_search=256,
    img_size_template=128,
    patch_size=16,

    # Architecture — tuned for ~5M params
    embed_dim=176,       # D: token dimension (default 256)
    state_dim=24,        # N: geodesic state on S^23 (default 64)
    expand_ratio=2,      # E = 352 (default: E = 512)
    num_blocks=5,        # L: depth (default 18 for LF-SSM-L)

    # Geodesic parameters (keep paper defaults)
    alpha=0.5,           # prior velocity weight
    eps=1e-6,            # numerical stability

    # Training parameters
    drop_path_rate=0.05, # lighter regularisation for smaller model

    # Loss weights (Eq. 13)
    lambda_iou=2.0,
    lambda_l1=5.0,
)

model = LFSSM(cfg_5m)

# ── Verify parameter count ────────────────────────────────────────
param_counts = model.get_param_count()
print("Parameter breakdown:")
for k, v in param_counts.items():
    print(f"  {k:20s}: {v:>10,d}  ({v/1e6:.2f}M)")

total = param_counts['total']
assert 4_500_000 < total < 5_500_000, f"Expected ~5M params, got {total/1e6:.2f}M"
print(f"\n✅ Total: {total/1e6:.2f}M parameters — within 5M target!")
```

### Other ~5M configurations you could try

```python
# Alternative A: Wider but shallower
cfg_5m_wide = ModelConfig(
    embed_dim=192, state_dim=24, num_blocks=4, expand_ratio=2,
)
# → ~5.18M params

# Alternative B: Narrower but deeper
cfg_5m_deep = ModelConfig(
    embed_dim=144, state_dim=48, num_blocks=8, expand_ratio=2,
)
# → ~4.86M params

# Alternative C: Balanced with larger manifold
cfg_5m_manifold = ModelConfig(
    embed_dim=176, state_dim=32, num_blocks=5, expand_ratio=2,
)
# → ~5.05M params
```

---

## 6. Training

### 6.0 Pre-training diagnostics (run this first!)

```python
# Cell 10a: Profile a single forward pass BEFORE training
# This reveals exactly where time is spent and catches issues early.

from lf_ssm.debug_utils import (
    profile_forward_pass, log_model_summary, log_tensor_stats,
    log_gpu_memory, check_gradients, DebugTimer, EpochLogger,
)
from lf_ssm.losses import TrackingLoss
import time, sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ── Model summary ─────────────────────────────────────────────────
log_model_summary(model)
log_gpu_memory()

# ── Profile one forward pass ──────────────────────────────────────
# Uses random data — no dataset needed. Shows per-block timings.
dummy_template = torch.randn(1, 3, 128, 128, device=device)
dummy_search   = torch.randn(1, 3, 256, 256, device=device)
timings = profile_forward_pass(model, dummy_template, dummy_search)

# ── Check output tensors ──────────────────────────────────────────
model.train()
cls_out, bbox_out = model(dummy_template, dummy_search)
log_tensor_stats("cls_score", cls_out)
log_tensor_stats("bbox_pred", bbox_out)

# ── Check gradient flow ──────────────────────────────────────────
loss = cls_out.mean() + bbox_out.mean()
loss.backward()
check_gradients(model)
log_gpu_memory()

print("\n✅ Pre-training diagnostics complete!")
print(f"   Estimated time per batch (batch_size=16): "
      f"~{timings['total'] * 16:.1f}s forward + similar backward")
print(f"   → ~{timings['total'] * 16 * 2:.1f}s total per batch")

del dummy_template, dummy_search, cls_out, bbox_out, loss
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 6.1 Training loop with verbose debugging

```python
# Cell 10b: Training functions with per-batch progress

# Optimizer (§6.1.3: AdamW, lr=1e-4, weight_decay=1e-4)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)

# Learning rate scheduler (cosine annealing)
num_epochs = 100  # Paper uses 300; reduce for Colab time constraints
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

# Loss function (Eq. 13)
criterion = TrackingLoss(
    lambda_iou=cfg_5m.lambda_iou,
    lambda_l1=cfg_5m.lambda_l1,
).to(device)

# ── How often to print batch progress ─────────────────────────────
PRINT_EVERY = 5  # Print every N batches (set to 1 for maximum verbosity)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """Train for one epoch with per-batch progress logging."""
    model.train()
    logger = EpochLogger(
        num_batches=len(loader), epoch=epoch, print_every=PRINT_EVERY,
    )

    for batch_idx, batch in enumerate(loader):
        batch_start = time.perf_counter()

        template = batch['template'].to(device)   # (B, 3, 128, 128)
        search   = batch['search'].to(device)      # (B, 3, 256, 256)
        bbox_gt  = batch['bbox'].to(device)         # (B, 4)
        cls_gt   = batch['cls_label'].to(device)    # (B, 1, H, W)

        # Forward
        cls_pred, bbox_pred = model(template, search)

        # Get center bbox from prediction map
        feat_size = bbox_pred.shape[-1]
        center = feat_size // 2
        bbox_center_pred = bbox_pred[:, :, center, center]  # (B, 4)

        losses = criterion(cls_pred, cls_gt, bbox_center_pred, bbox_gt)

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Log progress ──────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time = time.perf_counter() - batch_start

        loss_dict = {k: v.item() for k, v in losses.items()}
        logger.log_batch(batch_idx, batch_time, loss_dict)

        # Extra diagnostics on first batch of first epoch
        if batch_idx == 0 and epoch == 1:
            print("    ── First batch diagnostics ──")
            log_tensor_stats("    cls_pred", cls_pred)
            log_tensor_stats("    bbox_pred", bbox_pred)
            log_tensor_stats("    bbox_gt", bbox_gt)
            check_gradients(model, top_k=3)
            print("    ── End first batch diagnostics ──")

    return logger.summarise()


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    """Validate with progress logging."""
    model.eval()
    logger = EpochLogger(
        num_batches=len(loader), epoch=epoch, print_every=max(1, len(loader) // 3),
    )

    for batch_idx, batch in enumerate(loader):
        batch_start = time.perf_counter()

        template = batch['template'].to(device)
        search   = batch['search'].to(device)
        bbox_gt  = batch['bbox'].to(device)
        cls_gt   = batch['cls_label'].to(device)

        cls_pred, bbox_pred = model(template, search)
        feat_size = bbox_pred.shape[-1]
        center = feat_size // 2
        bbox_center_pred = bbox_pred[:, :, center, center]

        losses = criterion(cls_pred, cls_gt, bbox_center_pred, bbox_gt)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time = time.perf_counter() - batch_start

        loss_dict = {k: v.item() for k, v in losses.items()}
        logger.log_batch(batch_idx, batch_time, loss_dict)

    return logger.summarise()
```

### 6.2 Run training

```python
# Cell 11: Run training with full logging

best_val_loss = float('inf')
print("=" * 70)
print(f"STARTING TRAINING: {num_epochs} epochs, {len(train_loader)} batches/epoch")
print(f"Device: {device} | Batch size: {train_loader.batch_size}")
print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
print("=" * 70)

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"EPOCH {epoch}/{num_epochs}  (lr={optimizer.param_groups[0]['lr']:.2e})")
    print(f"{'─' * 70}")
    print("[Train]")
    train_losses = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch)

    # ── Validate ──────────────────────────────────────────────────
    print("[Val]")
    val_losses = validate(model, val_loader, criterion, device, epoch)

    scheduler.step()
    elapsed = time.time() - epoch_start
    lr = optimizer.param_groups[0]['lr']

    # ── Epoch summary ─────────────────────────────────────────────
    train_total = train_losses.get('total', 0)
    val_total = val_losses.get('total', 0)
    print(f"\n📊 Epoch {epoch:3d}/{num_epochs} SUMMARY: "
          f"Train={train_total:.4f} Val={val_total:.4f} "
          f"LR={lr:.2e} Time={elapsed:.1f}s")
    log_gpu_memory()

    # ── Save best model ───────────────────────────────────────────
    if val_total < best_val_loss:
        best_val_loss = val_total
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_total,
            'config': cfg_5m,
        }, '/content/best_lf_ssm_5m.pth')
        print(f"  ✅ Best model saved (val_loss={val_total:.4f})")

    # ── Periodic checkpoint ───────────────────────────────────────
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': cfg_5m,
        }, f'/content/checkpoint_epoch{epoch}.pth')
        print(f"  💾 Checkpoint saved: checkpoint_epoch{epoch}.pth")

    sys.stdout.flush()
```

---

## 7. Evaluation

```python
# Cell 12: Load best model and evaluate

checkpoint = torch.load('/content/best_lf_ssm_5m.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded best model from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
```

```python
# Cell 13: Evaluate on UAV123 sequences (Success/Precision metrics)

import numpy as np


def compute_iou(pred, gt):
    """Compute IoU between two boxes (x,y,w,h format)."""
    x1 = max(pred[0], gt[0])
    y1 = max(pred[1], gt[1])
    x2 = min(pred[0] + pred[2], gt[0] + gt[2])
    y2 = min(pred[1] + pred[3], gt[1] + gt[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = pred[2] * pred[3] + gt[2] * gt[3] - inter
    return inter / (union + 1e-7)


def compute_center_error(pred, gt):
    """Center location error in pixels."""
    pred_cx = pred[0] + pred[2] / 2
    pred_cy = pred[1] + pred[3] / 2
    gt_cx = gt[0] + gt[2] / 2
    gt_cy = gt[1] + gt[3] / 2
    return np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)


def track_sequence(model, seq, device, search_size=256, template_size=128,
                   search_factor=4.0, template_factor=2.0):
    """Run tracker on a full sequence."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frames = seq['frames']
    gt_bboxes = seq['bboxes']  # (N, 4) in x,y,w,h pixels

    # Initialise with first frame
    first_img = np.array(Image.open(frames[0]).convert('RGB'))
    init_bbox = gt_bboxes[0]

    # Create template from first frame
    cx, cy = init_bbox[0] + init_bbox[2]/2, init_bbox[1] + init_bbox[3]/2
    s = max(init_bbox[2], init_bbox[3]) * template_factor
    x1 = int(max(0, cx - s/2))
    y1 = int(max(0, cy - s/2))
    x2 = int(min(first_img.shape[1], cx + s/2))
    y2 = int(min(first_img.shape[0], cy + s/2))
    template_crop = Image.fromarray(first_img[y1:y2, x1:x2])
    template_crop = template_crop.resize((template_size, template_size), Image.BILINEAR)
    template_tensor = transform(template_crop).unsqueeze(0).to(device)

    model.set_template(template_tensor)

    predictions = [init_bbox.copy()]
    current_bbox = init_bbox.copy()

    for i in range(1, len(frames)):
        img = np.array(Image.open(frames[i]).convert('RGB'))
        cx = current_bbox[0] + current_bbox[2] / 2
        cy = current_bbox[1] + current_bbox[3] / 2
        s = max(current_bbox[2], current_bbox[3]) * search_factor

        # Crop search region
        sx1 = int(max(0, cx - s/2))
        sy1 = int(max(0, cy - s/2))
        sx2 = int(min(img.shape[1], cx + s/2))
        sy2 = int(min(img.shape[0], cy + s/2))
        search_crop = Image.fromarray(img[sy1:sy2, sx1:sx2])
        search_crop = search_crop.resize((search_size, search_size), Image.BILINEAR)
        search_tensor = transform(search_crop).unsqueeze(0).to(device)

        # Track
        cls_score, bbox_pred = model.track(search_tensor)

        # Get best location from classification score
        cls_map = cls_score[0, 0].cpu().numpy()
        max_idx = np.unravel_index(np.argmax(cls_map), cls_map.shape)
        feat_size = cls_map.shape[0]

        # Get bbox prediction at best location
        pred_bbox_norm = bbox_pred[0, :, max_idx[0], max_idx[1]].cpu().numpy()

        # Convert normalised bbox back to pixel coords in original image
        pred_cx = sx1 + pred_bbox_norm[0] * (sx2 - sx1)
        pred_cy = sy1 + pred_bbox_norm[1] * (sy2 - sy1)
        pred_w = pred_bbox_norm[2] * (sx2 - sx1)
        pred_h = pred_bbox_norm[3] * (sy2 - sy1)
        pred_bbox = np.array([pred_cx - pred_w/2, pred_cy - pred_h/2, pred_w, pred_h])

        predictions.append(pred_bbox)
        current_bbox = pred_bbox

    return np.array(predictions)


def evaluate_tracker(model, dataset_obj, device, num_sequences=None):
    """Evaluate on multiple sequences and compute metrics."""
    all_ious = []
    all_errors = []
    sequences = dataset_obj.sequences

    if num_sequences:
        sequences = sequences[:num_sequences]

    for seq in sequences:
        print(f"  Evaluating: {seq['name']} ({len(seq['frames'])} frames)...", end='')
        preds = track_sequence(model, seq, device)
        gt = seq['bboxes']

        ious = [compute_iou(preds[i], gt[i]) for i in range(len(preds))]
        errors = [compute_center_error(preds[i], gt[i]) for i in range(len(preds))]
        all_ious.extend(ious)
        all_errors.extend(errors)
        print(f" AUC={np.mean(ious):.3f}")

    all_ious = np.array(all_ious)
    all_errors = np.array(all_errors)

    # AUC: area under success plot (IoU thresholds 0 to 1)
    thresholds = np.arange(0, 1.05, 0.05)
    success_rates = [np.mean(all_ious >= t) for t in thresholds]
    auc = np.trapz(success_rates, thresholds)

    # Precision: % frames with center error < 20px
    precision = np.mean(all_errors < 20) * 100

    return {
        'AUC': auc,
        'Precision': precision,
        'mean_IoU': np.mean(all_ious),
    }
```

```python
# Cell 14: Run evaluation

print("=" * 60)
print("Evaluating LF-SSM (5M) on UAV123")
print("=" * 60)
metrics = evaluate_tracker(model, dataset, device, num_sequences=10)
print(f"\nResults:")
print(f"  AUC:       {metrics['AUC']:.4f}")
print(f"  Precision: {metrics['Precision']:.1f}%")
print(f"  Mean IoU:  {metrics['mean_IoU']:.4f}")
```

---

## 8. Inference on Video

```python
# Cell 15: Visualise tracking on a single sequence

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output


def visualise_tracking(model, seq, device, max_frames=100):
    """Visualise tracking results frame by frame."""
    preds = track_sequence(model, seq, device)
    gt = seq['bboxes']
    frames = seq['frames']

    n = min(len(frames), max_frames)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    step = max(1, n // 10)
    for idx, i in enumerate(range(0, n, step)):
        if idx >= 10:
            break
        img = Image.open(frames[i])
        axes[idx].imshow(img)

        # Ground truth (green)
        gt_rect = patches.Rectangle(
            (gt[i][0], gt[i][1]), gt[i][2], gt[i][3],
            linewidth=2, edgecolor='green', facecolor='none', label='GT'
        )
        axes[idx].add_patch(gt_rect)

        # Prediction (red)
        pred_rect = patches.Rectangle(
            (preds[i][0], preds[i][1]), preds[i][2], preds[i][3],
            linewidth=2, edgecolor='red', facecolor='none', label='Pred'
        )
        axes[idx].add_patch(pred_rect)
        axes[idx].set_title(f'Frame {i}')
        axes[idx].axis('off')

    axes[0].legend(loc='upper left')
    plt.suptitle(f"Tracking: {seq['name']}", fontsize=14)
    plt.tight_layout()
    plt.show()


# Visualise on first sequence
visualise_tracking(model, dataset.sequences[0], device)
```

---

## 9. Tips & Troubleshooting

### Colab-specific tips

| Issue | Solution |
|---|---|
| **Out of memory** | Reduce `batch_size` to 8 or 4 |
| **Colab disconnects** | Save checkpoints every 10 epochs; use Google Drive |
| **Slow training** | Use Colab Pro (A100/V100) for 3–5× speedup |
| **Dataset too large** | Use a subset: `dataset.samples = dataset.samples[:10000]` |

### Saving model to Drive

```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/best_lf_ssm_5m.pth /content/drive/MyDrive/
```

### Resuming training from checkpoint

```python
checkpoint = torch.load('/content/drive/MyDrive/checkpoint_epoch50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Adjusting the 5M config

```python
from lf_ssm import ModelConfig, LFSSM

# Even smaller (~3M)
cfg_3m = ModelConfig(embed_dim=144, state_dim=24, num_blocks=4, expand_ratio=2)

# Slightly larger (~8M)
cfg_8m = ModelConfig(embed_dim=192, state_dim=32, num_blocks=7, expand_ratio=2)

# Print parameter count for any config
model = LFSSM(cfg_3m)
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total/1e6:.2f}M")
```

### Key parameters reference

| Parameter | Where | Range | Notes |
|---|---|---|---|
| `embed_dim` | `ModelConfig` | 64–512 | Biggest impact on params (quadratic) |
| `state_dim` | `ModelConfig` | 16–256 | Manifold capacity; N=64 is paper default |
| `num_blocks` | `ModelConfig` | 2–24 | Linear impact on params |
| `expand_ratio` | `ModelConfig` | 1–4 | E = D × ratio; affects GSM width |
| `alpha` | `ModelConfig` | 0–1 | Prior velocity; 0=no momentum, 1=full |
| `patch_size` | `ModelConfig` | 8, 16, 32 | Smaller=more tokens (slower, finer) |
| `lr` | Optimizer | 1e-5–1e-3 | Paper uses 1e-4 |
| `batch_size` | DataLoader | 4–64 | Paper uses 32; reduce for Colab |
| `num_epochs` | Training loop | 50–500 | Paper uses 300 |
