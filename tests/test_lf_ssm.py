"""
Comprehensive tests for LF-SSM.

Run with:
    cd /home/allam/Desktop/AIC-4/LF_SSM
    python -m pytest tests/test_lf_ssm.py -v
"""

import pytest
import torch

from lf_ssm.config import ModelConfig, LF_SSM_S, LF_SSM_M, LF_SSM_L
from lf_ssm.geodesic_ops import (
    normalize_to_sphere,
    project_to_tangent_space,
    exponential_map,
    transport_prior_velocity,
    geodesic_state_evolution,
)
from lf_ssm.gsm_module import GeodesicStateModule
from lf_ssm.gsm_block import GSMBlock
from lf_ssm.patch_embed import PatchEmbed
from lf_ssm.prediction_head import PredictionHead
from lf_ssm.losses import FocalLoss, giou_loss, TrackingLoss
from lf_ssm.lf_ssm_model import LFSSM


# ── Helpers ────────────────────────────────────────────────────────

DEVICE = "cpu"
BATCH = 2
N = 32          # state dim (smaller for fast tests)
D = 64          # embed dim
E = 128         # expanded dim
T = 20          # sequence length


def _random_unit(shape):
    """Random unit vectors on S^{N-1}."""
    v = torch.randn(*shape)
    return v / v.norm(dim=-1, keepdim=True)


# ═══════════════════════════════════════════════════════════════════
#  1. Geodesic operations
# ═══════════════════════════════════════════════════════════════════

class TestNormalizeToSphere:
    def test_output_norm(self):
        h = torch.randn(BATCH, N)
        h_norm = normalize_to_sphere(h)
        norms = h_norm.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)

    def test_zero_vector_stability(self):
        """Should not produce NaN for near-zero input."""
        h = torch.zeros(N)
        h_norm = normalize_to_sphere(h, eps=1e-6)
        assert not torch.isnan(h_norm).any()


class TestProjectToTangentSpace:
    def test_orthogonality(self):
        """Projected vector must be orthogonal to base point: <v, h> ≈ 0."""
        h = _random_unit((BATCH, N))
        u = torch.randn(BATCH, N)
        v = project_to_tangent_space(u, h)
        dot = (v * h).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros(BATCH), atol=1e-5)

    def test_idempotence(self):
        """Projecting twice should give the same result."""
        h = _random_unit((BATCH, N))
        u = torch.randn(BATCH, N)
        v1 = project_to_tangent_space(u, h)
        v2 = project_to_tangent_space(v1, h)
        assert torch.allclose(v1, v2, atol=1e-5)


class TestExponentialMap:
    def test_norm_preservation(self):
        """After geodesic update, ‖h_new‖ = 1 (Eq. 6 proof)."""
        h = _random_unit((BATCH, N))
        u = torch.randn(BATCH, N)
        v = project_to_tangent_space(u, h)
        delta = torch.rand(BATCH, 1) * 0.5

        h_new = exponential_map(h, v, delta)
        norms = h_new.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH), atol=1e-4)

    def test_zero_velocity(self):
        """With zero tangent vector, the state should not move."""
        h = _random_unit((BATCH, N))
        v = torch.zeros(BATCH, N)
        delta = torch.ones(BATCH, 1)
        h_new = exponential_map(h, v, delta)
        assert torch.allclose(h, h_new, atol=1e-5)


class TestTransportPriorVelocity:
    def test_stays_in_tangent_space(self):
        """Transported prior must lie in T_{h_new} S^{N-1}."""
        h_new = _random_unit((BATCH, N))
        v = torch.randn(BATCH, N)
        v_prior = transport_prior_velocity(v, h_new)
        dot = (v_prior * h_new).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros(BATCH), atol=1e-5)


class TestGeodesicStateEvolution:
    def test_output_shape(self):
        x_seq = torch.randn(BATCH, T, E)
        B_proj = torch.nn.Linear(E, N)
        delta_proj = torch.nn.Linear(E, 1)
        h0 = normalize_to_sphere(torch.randn(1, N))

        H = geodesic_state_evolution(x_seq, B_proj, delta_proj, h0, alpha=0.5)
        assert H.shape == (BATCH, T, N)

    def test_all_states_on_sphere(self):
        """Every hidden state must have unit norm."""
        x_seq = torch.randn(BATCH, T, E)
        B_proj = torch.nn.Linear(E, N)
        delta_proj = torch.nn.Linear(E, 1)
        h0 = normalize_to_sphere(torch.randn(1, N))

        H = geodesic_state_evolution(x_seq, B_proj, delta_proj, h0, alpha=0.5)
        norms = H.norm(dim=-1)  # (B, T)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


# ═══════════════════════════════════════════════════════════════════
#  2. Module shape tests
# ═══════════════════════════════════════════════════════════════════

class TestGSMModule:
    def test_output_shape(self):
        gsm = GeodesicStateModule(
            embed_dim=D, state_dim=N, expand_ratio=2, conv_kernel_size=3,
        )
        x = torch.randn(BATCH, T, D)
        y = gsm(x)
        assert y.shape == (BATCH, T, D)


class TestGSMBlock:
    def test_residual_shape(self):
        block = GSMBlock(
            embed_dim=D, state_dim=N, expand_ratio=2, conv_kernel_size=3,
        )
        z = torch.randn(BATCH, T, D)
        z_out = block(z)
        assert z_out.shape == (BATCH, T, D)


class TestPatchEmbed:
    def test_search_tokens(self):
        pe = PatchEmbed(img_size=256, patch_size=16, in_channels=3, embed_dim=D)
        x = torch.randn(BATCH, 3, 256, 256)
        tokens = pe(x)
        assert tokens.shape == (BATCH, 256, D)

    def test_template_tokens(self):
        pe = PatchEmbed(img_size=128, patch_size=16, in_channels=3, embed_dim=D)
        x = torch.randn(BATCH, 3, 128, 128)
        tokens = pe(x)
        assert tokens.shape == (BATCH, 64, D)


class TestPredictionHead:
    def test_output_shapes(self):
        head = PredictionHead(embed_dim=D, num_classes=1)
        x = torch.randn(BATCH, D, 16, 16)
        cls_score, bbox_pred = head(x)
        assert cls_score.shape == (BATCH, 1, 16, 16)
        assert bbox_pred.shape == (BATCH, 4, 16, 16)


# ═══════════════════════════════════════════════════════════════════
#  3. Loss tests
# ═══════════════════════════════════════════════════════════════════

class TestLosses:
    def test_focal_loss_positive(self):
        fl = FocalLoss()
        pred = torch.randn(BATCH, 1, 4, 4)
        target = torch.ones(BATCH, 1, 4, 4)
        loss = fl(pred, target)
        assert loss.item() >= 0

    def test_giou_loss_range(self):
        pred = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
        tgt  = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        loss = giou_loss(pred, tgt)
        assert 0 <= loss.item() <= 2

    def test_tracking_loss_returns_dict(self):
        tl = TrackingLoss()
        cls_p = torch.randn(BATCH, 1, 4, 4)
        cls_t = torch.ones(BATCH, 1, 4, 4)
        box_p = torch.rand(BATCH, 4)
        box_t = torch.rand(BATCH, 4)
        out = tl(cls_p, cls_t, box_p, box_t)
        assert set(out.keys()) == {"total", "cls", "giou", "l1"}


# ═══════════════════════════════════════════════════════════════════
#  4. Full model tests
# ═══════════════════════════════════════════════════════════════════

def _small_cfg():
    """Tiny config for fast testing."""
    return ModelConfig(
        img_size_search=64,
        img_size_template=32,
        patch_size=8,
        embed_dim=32,
        state_dim=16,
        expand_ratio=2,
        num_blocks=2,
        drop_path_rate=0.0,
    )


class TestLFSSMModel:
    def test_forward_shape(self):
        cfg = _small_cfg()
        model = LFSSM(cfg)
        template = torch.randn(BATCH, 3, 32, 32)
        search   = torch.randn(BATCH, 3, 64, 64)
        cls, bbox = model(template, search)
        fs = cfg.search_feat_size  # 64/8 = 8
        assert cls.shape == (BATCH, 1, fs, fs)
        assert bbox.shape == (BATCH, 4, fs, fs)

    def test_template_caching(self):
        cfg = _small_cfg()
        model = LFSSM(cfg)
        model.eval()
        template = torch.randn(1, 3, 32, 32)
        search   = torch.randn(1, 3, 64, 64)

        model.set_template(template)
        cls, bbox = model.track(search)

        fs = cfg.search_feat_size
        assert cls.shape == (1, 1, fs, fs)
        assert bbox.shape == (1, 4, fs, fs)

    def test_gradient_flow(self):
        """Ensure gradients flow back through the geodesic operations."""
        cfg = _small_cfg()
        model = LFSSM(cfg)
        template = torch.randn(BATCH, 3, 32, 32)
        search   = torch.randn(BATCH, 3, 64, 64)
        cls, bbox = model(template, search)
        loss = cls.mean() + bbox.mean()
        loss.backward()
        # Check that at least the first block has gradients
        for name, p in model.blocks[0].named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                break


class TestModelVariants:
    @pytest.mark.parametrize("factory", [LF_SSM_S, LF_SSM_M, LF_SSM_L])
    def test_variant_instantiation(self, factory):
        cfg = factory()
        model = LFSSM(cfg)
        params = model.get_param_count()
        assert params["total"] > 0
        # Check num_blocks matches
        assert len(model.blocks) == cfg.num_blocks

    def test_param_count_ordering(self):
        """S < M < L in parameter count."""
        s = LFSSM(LF_SSM_S()).get_param_count()["total"]
        m = LFSSM(LF_SSM_M()).get_param_count()["total"]
        l = LFSSM(LF_SSM_L()).get_param_count()["total"]
        assert s < m < l

    def test_custom_config(self):
        """User can create arbitrary custom configs."""
        cfg = ModelConfig(
            img_size_search=128, img_size_template=64,
            patch_size=8, embed_dim=128, state_dim=32,
            expand_ratio=3, num_blocks=4,
        )
        model = LFSSM(cfg)
        template = torch.randn(1, 3, 64, 64)
        search   = torch.randn(1, 3, 128, 128)
        cls, bbox = model(template, search)
        assert cls.shape[0] == 1
        assert bbox.shape[0] == 1
