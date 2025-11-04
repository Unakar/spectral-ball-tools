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


## NOTE: 彻底移除在 base 中的奇异向量求解，转移到 kernels/power_iteration.py


# -----------------------------------------------------------------------------
# Objective helpers (decoupled, solver-agnostic)
#   - compute_phi(G, Θ, λ): Φ(λ) = msign(G + λΘ)
#   - compute_f(G, Θ, λ): f(λ) = <Θ, Φ(λ)>
#   Legacy convenience evaluate_objective_and_stats kept for compatibility.
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_phi(
    G: torch.Tensor,
    Theta: torch.Tensor,
    lambda_value: float,
    msign_steps: int = 5,
) -> torch.Tensor:
    """Compute Φ(λ) = msign(G + λΘ) on GPU (no SVD)."""
    device = G.device
    lambda_tensor = torch.tensor(lambda_value, device=device, dtype=torch.float32)
    Z = G + lambda_tensor * Theta
    return msign(Z, steps=msign_steps)


@torch.no_grad()
def compute_f(
    G: torch.Tensor,
    Theta: torch.Tensor,
    lambda_value: float,
    msign_steps: int = 5,
) -> float:
    """Compute scalar f(λ) = <Θ, Φ(λ)> with Φ(λ)=msign(G+λΘ)."""
    Phi = compute_phi(G, Theta, lambda_value, msign_steps)
    return float(inner_product(Theta, Phi).item())


@torch.no_grad()
def evaluate_objective_and_stats(
    G: torch.Tensor,
    Theta: torch.Tensor,
    lambda_value: float,
    msign_steps: int = 5,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy helper retained for compatibility.
    Returns (f, Φ, X=ΘᵀΦ, q=ZᵀΦ) where Z=G+λΘ.
    """
    device = G.device
    lambda_tensor = torch.tensor(lambda_value, device=device, dtype=torch.float32)
    Z = G + lambda_tensor * Theta
    Phi = msign(Z, steps=msign_steps)
    X = Theta.mT @ Phi
    q = Z.mT @ Phi
    f_val = float(inner_product(Theta, Phi).item())
    return f_val, Phi, X, q



# -----------------------------------------------------------------------------
# Standard result record
# -----------------------------------------------------------------------------
@dataclass
class SolverResult:
    """统一的求解结果结构。

    - method: 求解器名称（'brent'/'bisection'/'secant'/'fixed_point'/'newton'）。
    - solution: 最终 λ。
    - residual: 最终 |f(λ)|。
    - iterations: 迭代步数（唯一关注的计数指标）。
    - converged: 是否满足收敛判据（各 solver 中定义）。
    - time_sec: 求解耗时（秒）。
    - bracket: 可选，括号区间 (lo, hi)。
    - history: 可选，包含 'solution' 与 'residual' 的轨迹。
    """
    method: str
    solution: float
    residual: float
    iterations: int
    converged: bool
    time_sec: float = 0.0
    bracket: tuple[float, float] | None = None
    history: Dict[str, Any] | None = None


## NOTE: 移除 generate_test_matrices，避免与 root_solver 的数据构造重复
