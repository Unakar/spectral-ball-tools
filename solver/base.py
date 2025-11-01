# solver/common.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import torch
from kernels.msign import msign


# -----------------------------------------------------------------------------
# Basic utilities (FP32 accumulation to reduce mixed-precision error)
# -----------------------------------------------------------------------------
@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product <a, b>, returned as a scalar tensor on GPU."""
    return (a.to(torch.float32) * b.to(torch.float32)).sum()

@torch.no_grad()
def trace_fp32(a: torch.Tensor) -> torch.Tensor:
    """Trace with FP32 accumulation; returns a scalar tensor on GPU."""
    return torch.trace(a.to(torch.float32))


# -----------------------------------------------------------------------------
# Top singular vectors via bilateral power iteration (SVD-free, GPU-friendly)
# Iteration:
#   1) u ← normalize(W v)
#   2) v ← normalize(W^T u)
# Rayleigh quotient approximates σ_max:  σ ≈ u^T (W v)
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_top_singular_vectors(
    W: torch.Tensor,
    iters: int = 3,
    tol: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns (u1, v1, sigma_est). All computation stays on GPU.
    Normalization is done in FP32; only GEMV/GEMM are used.
    """
    assert W.is_cuda, "Expect CUDA tensors."
    n, m = W.shape

    v = torch.randn(m, device=W.device, dtype=torch.float32)
    v = v / (v.norm() + 1e-20)

    sigma_prev = 0.0
    for _ in range(iters):
        u = W @ v
        u_norm = u.norm()
        if u_norm < 1e-30:
            u.zero_(); v.zero_()
            return u, v, 0.0
        u = u / u_norm

        v = W.mT @ u
        v_norm = v.norm()
        if v_norm < 1e-30:
            u.zero_(); v.zero_()
            return u, v, 0.0
        v = v / v_norm

        sigma = torch.dot(u, W @ v).item()
        if abs(sigma - sigma_prev) < tol * max(1.0, abs(sigma_prev)):
            break
        sigma_prev = sigma

    return u, v, float(sigma_prev)


# -----------------------------------------------------------------------------
# Unified objective evaluation:
#   f(λ) = <Θ, Φ(λ)>, where Φ(λ) = msign(G + λΘ)
#   Returns (f, Φ, X, q), where:
#     X = Θ^T Φ
#     q = (G + λΘ)^T Φ  == (Z^T Z)^{1/2}  (avoids explicit matrix sqrt/inv)
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_objective_and_stats(
    G: torch.Tensor,
    Theta: torch.Tensor,
    lambda_value: float,
    msign_steps: int = 5,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = G.device
    lambda_tensor = torch.tensor(lambda_value, device=device, dtype=torch.float32)
    Z = G + lambda_tensor * Theta
    Phi = msign(Z, steps=msign_steps)
    X = Theta.mT @ Phi
    q = Z.mT @ Phi
    f_val = float(inner_product(Theta, Phi).item())
    return f_val, Phi, X, q


# -----------------------------------------------------------------------------
# Fixed-point step size (SVD-free):
#   c = ||Z||_* / (m * ||Θ||_F^2), and  ||Z||_* = tr((Z^T Z)^{1/2}) = tr(q)
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_step_size_via_q(
    Z: torch.Tensor,
    Theta: torch.Tensor,
    Phi: torch.Tensor,
) -> float:
    m = Z.shape[1]
    trace_q = float(trace_fp32(Z.mT @ Phi).item())
    theta_fro_sq = float(inner_product(Theta, Theta).item())
    denom = max(1e-12, m * theta_fro_sq)
    return trace_q / denom


# -----------------------------------------------------------------------------
# Standard result record
# -----------------------------------------------------------------------------
@dataclass
class SolverResult:
    method: str
    solution: float
    residual: float
    iterations: int
    converged: bool
    time_sec: float = 0.0
    function_evaluations: int = 0
    bracket: tuple[float, float] | None = None
    history: Dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# Random test matrices (GPU-only)
# Θ is constructed from top singular vectors of a random W for realism
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_test_matrices(rows: int, cols: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    assert torch.cuda.is_available(), "CUDA not available; GPU is required."
    device = torch.device("cuda")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    W = torch.randn(rows, cols, device=device, dtype=torch.float32) / math.sqrt(cols)
    G = torch.randn(rows, cols, device=device, dtype=torch.float32) / math.sqrt(cols)
    u1, v1, _ = compute_top_singular_vectors(W, iters=3, tol=1e-6)
    Theta = torch.outer(u1, v1)  # Θ = u1 v1^T (gradient of spectral norm at W)
    return G, Theta
