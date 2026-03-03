"""
Loss functions for LF-SSM training (Eq. 13).

    L = L_cls  +  λ_iou · L_iou  +  λ_L1 · L_L1

- L_cls  :  Focal Loss for foreground / background classification.
- L_iou  :  Generalised IoU (GIoU) loss for bounding-box regression.
- L_L1   :  L1 loss for coordinate regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Focal Loss (classification) ────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for dense classification (Lin et al., 2017).

    Parameters
    ----------
    alpha : float
        Balancing factor for positive / negative examples. Default 0.25.
    gamma : float
        Focusing parameter that down-weights easy examples. Default 2.0.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        pred : Tensor, shape (B, 1, H, W)
            Raw logits (before sigmoid).
        target : Tensor, shape (B, 1, H, W)
            Ground-truth labels ∈ {0, 1}.
        """
        prob = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


# ── GIoU Loss (bounding-box regression) ────────────────────────────

def _box_xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def giou_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Generalised Intersection-over-Union loss.

    Parameters
    ----------
    pred : Tensor, shape (N, 4)
        Predicted boxes in (cx, cy, w, h) normalised coordinates.
    target : Tensor, shape (N, 4)
        Ground-truth boxes in (cx, cy, w, h) normalised coordinates.

    Returns
    -------
    Tensor (scalar)
        Mean GIoU loss ∈ [0, 2].
    """
    pred_xyxy = _box_xywh_to_xyxy(pred)
    tgt_xyxy = _box_xywh_to_xyxy(target)

    # Intersection
    inter_x1 = torch.max(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    inter_y1 = torch.max(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    inter_x2 = torch.min(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    inter_y2 = torch.min(pred_xyxy[..., 3], tgt_xyxy[..., 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    area_pred = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0) * \
                (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=0)
    area_tgt  = (tgt_xyxy[..., 2] - tgt_xyxy[..., 0]).clamp(min=0) * \
                (tgt_xyxy[..., 3] - tgt_xyxy[..., 1]).clamp(min=0)
    union = area_pred + area_tgt - inter_area + 1e-7

    iou = inter_area / union

    # Enclosing box
    enc_x1 = torch.min(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    enc_y1 = torch.min(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    enc_x2 = torch.max(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    enc_y2 = torch.max(pred_xyxy[..., 3], tgt_xyxy[..., 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-7

    giou = iou - (enc_area - union) / enc_area
    return (1 - giou).mean()


# ── Combined tracking loss (Eq. 13) ────────────────────────────────

class TrackingLoss(nn.Module):
    """Combined loss:  L = L_cls + λ_iou · L_iou + λ_L1 · L_L1.

    Parameters
    ----------
    lambda_iou : float
        Weight for GIoU loss. Default 2.0.
    lambda_l1 : float
        Weight for L1 loss. Default 5.0.
    focal_alpha : float
        Focal loss alpha. Default 0.25.
    focal_gamma : float
        Focal loss gamma. Default 2.0.
    """

    def __init__(
        self,
        lambda_iou: float = 2.0,
        lambda_l1: float = 5.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_iou = lambda_iou
        self.lambda_l1 = lambda_l1
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        cls_pred: Tensor,
        cls_target: Tensor,
        bbox_pred: Tensor,
        bbox_target: Tensor,
    ) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        cls_pred : Tensor, shape (B, 1, H, W)
            Classification logits.
        cls_target : Tensor, shape (B, 1, H, W)
            Ground-truth class labels.
        bbox_pred : Tensor, shape (N, 4)
            Predicted bounding boxes (cx, cy, w, h) for positive samples.
        bbox_target : Tensor, shape (N, 4)
            Ground-truth bounding boxes for positive samples.

        Returns
        -------
        dict with keys 'total', 'cls', 'giou', 'l1'.
        """
        l_cls = self.focal_loss(cls_pred, cls_target)
        l_iou = giou_loss(bbox_pred, bbox_target)
        l_l1 = F.l1_loss(bbox_pred, bbox_target)

        total = l_cls + self.lambda_iou * l_iou + self.lambda_l1 * l_l1
        return {
            "total": total,
            "cls": l_cls,
            "giou": l_iou,
            "l1": l_l1,
        }
