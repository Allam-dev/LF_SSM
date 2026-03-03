"""
Core geodesic operations on the unit sphere S^{N-1}.

These implement the Riemannian geometry primitives from §3.2 and §4.2 of the
LF-SSM paper.  All functions operate on batched tensors and are fully
differentiable through standard PyTorch autograd.

Mathematical summary
====================
- **Unit sphere**:  S^{N-1} = { h ∈ R^N : ‖h‖ = 1 }  (Eq. 3)
- **Tangent space projection**:  Π_h(u) = u − ⟨u, h⟩ h  (Eq. 4)
- **Exponential map**:
      Exp_h(Δv) = cos(Δ‖v‖) · h + sin(Δ‖v‖) · v / (‖v‖ + ε)  (Eq. 5)
- **Prior velocity transport**:
      v^{prior}_{t+1} = v_t − ⟨v_t, h_t⟩ h_t  (Eq. 7)
"""

import torch
from torch import Tensor


def normalize_to_sphere(h: Tensor, eps: float = 1e-6) -> Tensor:
    """Project a vector onto the unit sphere S^{N-1}.

    Implements the initial state normalization from Eq. 9:
        h₀ = ĥ₀ / (‖ĥ₀‖ + ε)

    Parameters
    ----------
    h : Tensor, shape (..., N)
        Arbitrary vector(s) to normalise.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    Tensor, shape (..., N)
        Unit vector(s) on S^{N-1}.
    """
    return h / (h.norm(dim=-1, keepdim=True) + eps)


def project_to_tangent_space(u: Tensor, h: Tensor) -> Tensor:
    """Orthogonal projection onto the tangent space T_h S^{N-1}.

    Implements Eq. 4:
        v = Π_h(u) = u − ⟨u, h⟩ h

    The result satisfies ⟨v, h⟩ = 0  (orthogonality condition).

    Parameters
    ----------
    u : Tensor, shape (..., N)
        Vector(s) to project.
    h : Tensor, shape (..., N)
        Base point(s) on the unit sphere.

    Returns
    -------
    Tensor, shape (..., N)
        Tangent vector(s) in T_h S^{N-1}.
    """
    # ⟨u, h⟩  – inner product along the last dimension
    coeff = (u * h).sum(dim=-1, keepdim=True)  # (..., 1)
    return u - coeff * h


def exponential_map(
    h: Tensor, v: Tensor, delta: Tensor, eps: float = 1e-6
) -> Tensor:
    """Move along the geodesic on S^{N-1} via the exponential map.

    Implements Eq. 5:
        h_new = cos(Δ‖v‖) · h + sin(Δ‖v‖) · v / (‖v‖ + ε)

    The product Δ‖v‖ is the arc-length traveled on the sphere.

    Parameters
    ----------
    h : Tensor, shape (..., N)
        Current state on S^{N-1}.
    v : Tensor, shape (..., N)
        Tangent vector in T_h S^{N-1}.
    delta : Tensor, shape (..., 1) or broadcastable
        Input-dependent step size Δ > 0.
    eps : float
        Numerical stability constant (paper uses 1e-6).

    Returns
    -------
    Tensor, shape (..., N)
        New state h_new on S^{N-1}.
    """
    v_norm = v.norm(dim=-1, keepdim=True)          # ‖v‖,  (..., 1)
    angle = delta * v_norm                          # Δ‖v‖, arc-length

    h_new = torch.cos(angle) * h + torch.sin(angle) * v / (v_norm + eps)
    return h_new


def transport_prior_velocity(v: Tensor, h_new: Tensor) -> Tensor:
    """Transport the velocity vector to the tangent space at the new state.

    Implements Eq. 7 (prior velocity mechanism):
        v^{prior}_{t+1} = v_t − ⟨v_t, h_t⟩ h_t

    This is geometrically an approximation of parallel transport on S^{N-1}
    via orthogonal projection.  It implements *geometric forgetting*: when the
    trajectory curves sharply (rapid appearance change), most of the prior is
    discarded; when smooth, most is retained (§7.2).

    Parameters
    ----------
    v : Tensor, shape (..., N)
        Velocity vector from the previous step.
    h_new : Tensor, shape (..., N)
        New state on S^{N-1} (the transport destination).

    Returns
    -------
    Tensor, shape (..., N)
        Transported prior velocity in T_{h_new} S^{N-1}.
    """
    return project_to_tangent_space(v, h_new)


def geodesic_state_evolution(
    x_seq: Tensor,
    B_proj: "torch.nn.Linear",
    delta_proj: "torch.nn.Linear",
    h0: Tensor,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    """Run the full geodesic state evolution over a token sequence.

    Implements Algorithm 1 from the paper.

    Parameters
    ----------
    x_seq : Tensor, shape (B, T, E)
        Expanded input features (after linear expansion + conv + SiLU).
    B_proj : nn.Linear
        Linear_B : E → N  (input projection, Eq. 9).
    delta_proj : nn.Linear
        Linear_Δ : E → 1  (step size projection, Eq. 9).
    h0 : Tensor, shape (N,) or (1, N)
        Learnable initial state (already normalised to S^{N-1}).
    alpha : float
        Confidence weight for prior velocity (Algorithm 1, line 3).
    eps : float
        Stability constant ε.

    Returns
    -------
    Tensor, shape (B, T, N)
        Collected hidden states H = [h₁, h₂, …, h_T].
    """
    B_batch, T, E = x_seq.shape
    N = h0.shape[-1]
    device = x_seq.device

    # Broadcast h0 to batch:  (1, N) → (B, N)
    ht = h0.expand(B_batch, -1).clone()  # current state on S^{N-1}

    # Prior velocity initialised to zero  (Algorithm 1, line 1)
    v_prior = torch.zeros(B_batch, N, device=device)

    states = []

    for t in range(T):
        xt = x_seq[:, t, :]                         # (B, E)

        # ── Input projection (Eq. 9)  ──────────────────────────────
        ut = B_proj(xt)                              # (B, N)

        # ── Combine with prior momentum (Algorithm 1, line 3) ─────
        wt = ut + alpha * v_prior                    # (B, N)

        # ── Tangent space projection (Eq. 4 / Alg 1(4)) ───────────
        vt = project_to_tangent_space(wt, ht)        # (B, N)

        # ── Input-dependent step size (Eq. 9 / Alg 1(5)) ──────────
        delta_t = torch.nn.functional.softplus(
            delta_proj(xt)
        )                                            # (B, 1)

        # ── Geodesic update (Eq. 5 / Alg 1(6)) ────────────────────
        ht = exponential_map(ht, vt, delta_t, eps)   # (B, N)

        # ── Transport prior to new tangent space (Eq. 7 / Alg 1(7))
        v_prior = transport_prior_velocity(vt, ht)   # (B, N)

        states.append(ht)

    # Stack along time:  list of (B, N) → (B, T, N)
    return torch.stack(states, dim=1)
