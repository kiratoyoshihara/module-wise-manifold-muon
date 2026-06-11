import math
import torch
from msign import msign


# Gram-space projections (Tilde Research, 2025)
# The projector P determines which manifold we optimize on.
# All three choices are self-adjoint, so the dual ascent framework
# from Bernstein (2025) applies unchanged i.e., only P differs.
def gram_project(M, manifold):
    """
    Apply self-adjoint projector P to a symmetric matrix.
    Supports 2D (n, n) and 3D (B, n, n).

    Stiefel:  P = Id    →  constrain W^T W = I
    DGram:    P = Off   →  constrain off-diag(W^T W) = 0
    Oblique:  P = Diag  →  constrain diag(W^T W) = I
    """
    if manifold == "stiefel":
        return M
    n = M.shape[-1]
    eye = torch.eye(n, device=M.device, dtype=M.dtype)
    if M.ndim == 3:
        eye = eye.unsqueeze(0)
    if manifold == "dgram":
        return M * (1 - eye)
    elif manifold == "oblique":
        return M * eye
    else:
        raise ValueError(f"Unknown manifold: {manifold}")


def retract(W, manifold):
    """
    Retract W back onto the specified manifold.
    Supports 2D (m, n) and 3D (B, m, n). Assumes tall matrices (m >= n).

    Stiefel:  nearest orthogonal matrix via polar factor (msign)
    DGram:    polar factor scaled by singular values (SVD-based)
              W_ret = UV^T diag(σ),  so  W_ret^T W_ret = diag(σ^2)
    Oblique:  normalize each column to unit norm
    """
    if manifold == "stiefel":
        return msign(W)
    elif manifold == "dgram":
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        return (U @ Vt) * S.unsqueeze(-2)
    elif manifold == "oblique":
        return W / torch.norm(W, dim=-2, keepdim=True).clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown manifold: {manifold}")


def init_on_manifold(W, manifold):
    """
    Project W onto the manifold before training begins.
    Handles both tall and wide matrices.
    """
    should_transpose = W.shape[-2] < W.shape[-1]
    if should_transpose:
        W = W.mT
    W = retract(W, manifold)
    if should_transpose:
        W = W.mT
    return W


@torch.no_grad()
def manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6, manifold="stiefel"):
    """
    Manifold Muon update for a single weight matrix.

    Solves the tangent-space-constrained steepest descent problem:
      min_A  tr(G^T A)
      s.t.   ||A||_spectral <= eta,  P(A^T W + W^T A) = 0

    via dual ascent, then retracts to the manifold.

    Args:
        W: (m, n) weight matrix, assumed on the manifold
        G: (m, n) gradient or momentum
        eta: spectral norm bound on the update
        alpha: dual ascent learning rate
        steps: max dual ascent iterations
        tol: RMS convergence tolerance for tangent violation
        manifold: "stiefel", "dgram", or "oblique"
    """
    should_transpose = W.shape[-2] < W.shape[-1]
    if should_transpose:
        W = W.mT
        G = G.mT

    P = lambda M: gram_project(M, manifold)

    Lambda = -0.25 * P(W.mT @ G + G.mT @ W)

    for step in range(steps):
        sym_Lambda = 0.5 * (Lambda + Lambda.mT)
        A = msign(G + 2 * W @ P(sym_Lambda))
        H = P(W.mT @ A + A.mT @ W)

        if torch.norm(H) / math.sqrt(H.numel()) < tol:
            break

        Lambda -= alpha * (1 - step / steps) * H

    new_W = W - eta * A
    new_W = retract(new_W, manifold)

    return new_W.mT if should_transpose else new_W


@torch.no_grad()
def batched_manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, manifold="stiefel"):
    """
    Manifold Muon update for a bank of weight matrices.

    Identical to manifold_muon but operates on (B, m, n) tensors,
    processing all matrices in a transformer parameter bank in parallel.
    Uses fixed iteration count since per-matrix early stopping is
    not practical in the batched setting.

    Args:
        W: (B, m, n) weight matrices, each on the manifold
        G: (B, m, n) gradients or momentum
        eta: spectral norm bound on the update
        alpha: dual ascent learning rate
        steps: dual ascent iterations (fixed)
        manifold: "stiefel", "dgram", or "oblique"
    """
    should_transpose = W.shape[-2] < W.shape[-1]
    if should_transpose:
        W = W.mT
        G = G.mT

    P = lambda M: gram_project(M, manifold)

    # Lambda: (B, n, n)
    Lambda = -0.25 * P(W.mT @ G + G.mT @ W)

    for step in range(steps):
        sym_Lambda = 0.5 * (Lambda + Lambda.mT)
        A = msign(G + 2 * W @ P(sym_Lambda))
        H = P(W.mT @ A + A.mT @ W)
        Lambda -= alpha * (1 - step / steps) * H

    new_W = W - eta * A
    new_W = retract(new_W, manifold)

    return new_W.mT if should_transpose else new_W