# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass
from typing import Dict, Any

import torch
from kernels.msign import msign  # High-precision / efficient Polar Express msign implementation
from .base import SolverResult, inner_product, trace_fp32


"""
固定点法：求解 r(λ)=tr(ΘᵀΦ(λ))=0，其中 Φ(λ)=msign(G+λΘ)。

更新公式（SVD-free，利用 q=ZᵀΦ=(ZᵀZ)^{1/2}）：
  λ ← [ tr(X q) − tr(X)·tr(q)/m − tr(ΘᵀG) ] / tr(ΘᵀΘ)，
  其中 Z=G+λΘ, Φ=msign(Z), X=ΘᵀΦ, q=ZᵀΦ。

收敛判据（仅函数值）：
  - 仅当 |r(λ)| ≤ tolerance_f 判定收敛；否则达到迭代上限视为未收敛。
"""


# -----------------------------------------------------------------------------
# Fixed-point solver（实现见 solve_with_fixed_point）
# 关键恒等式：q=zᵀΦ=(zᵀz)^{1/2}，可避免显式矩阵平方根/逆
# 更新：λ ← [ tr(X q) − tr(X)·tr(q)/m − tr(ΘᵀG) ] / tr(ΘᵀΘ)
# -----------------------------------------------------------------------------


@torch.no_grad()
def solve_with_fixed_point(
    G: torch.Tensor,
    Theta: torch.Tensor,
    tolerance_f: float = 1e-6,
    tolerance_x: float = 1e-8,
    max_iterations: int = 50,
    msign_steps: int = 5,
) -> SolverResult:
    start = time.perf_counter()
    assert G.is_cuda and Theta.is_cuda, "Expected CUDA tensors."
    n, m = G.shape

    tr_Th_Th = trace_fp32(Theta.mT @ Theta)  # tr(ΘᵀΘ)
    tr_Th_G  = trace_fp32(Theta.mT @ G)      # tr(ΘᵀG)

    # Initialize λ = - tr(ΘᵀG) / tr(ΘᵀΘ)
    lam = -tr_Th_G / (tr_Th_Th + 1e-30)
    prev_lam = None

    history: Dict[str, Any] = {"solution": [], "residual": []}

    for it in range(1, max_iterations + 1):
        Z = G + lam * Theta
        Phi = msign(Z, steps=msign_steps)
        X = Theta.mT @ Phi
        q = Z.mT @ Phi

        tr_X  = trace_fp32(X)
        tr_q  = trace_fp32(q)
        tr_Xq = trace_fp32(X @ q)
        f_val = tr_X
        f_abs = float(f_val.abs().item())
        history["solution"].append(float(lam.item()))
        history["residual"].append(float(f_abs))

        if f_abs <= tolerance_f:
            return SolverResult("fixed_point", float(lam.item()), f_abs, it, True,
                                time.perf_counter() - start, history=history)

        # Fixed-point update
        numerator = tr_Xq - tr_X * tr_q / m - tr_Th_G
        prev_lam = lam
        lam = numerator / (tr_Th_Th + 1e-30)

    # Max iterations reached
    return SolverResult("fixed_point", float(lam.item()), f_abs, max_iterations, False,
                        time.perf_counter() - start, history=history)
