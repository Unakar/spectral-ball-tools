# solver/newton.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Dict, Any

import torch
from kernels.msign import msign
from kernels.dmsign import dmsign

from .base import (
    SolverResult,
    inner_product,
    compute_f,
)

"""
Newton 法：求解 f(λ)=⟨Θ, Φ(λ)⟩=0，Φ(λ)=msign(G+λΘ)。

导数：
  优先使用 dmsign 的解析 VJP：f'(λ)=⟨Θ, dΦ/dA[Θ]⟩，其中 A=G+λΘ；
  若不可用，则回退中心差分近似。

收敛判据（仅函数值）：
  - 仅当 |f(λ)| ≤ tolerance_f 判定收敛；否则达到迭代上限视为未收敛。

统计：仅统计迭代步数 iterations；不统计函数评估次数。
"""
@torch.no_grad()
def solve_with_newton(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    tolerance_f: float = 1e-10,
    max_iterations: int = 50,
    msign_steps: int = 5,
    use_numeric_derivative: bool = False,
    finite_diff_eps: float = 1e-4,
) -> SolverResult:
    start = time.perf_counter()
    lambda_value = initial_guess
    f = compute_f(G, Theta, lambda_value, msign_steps)

    history: Dict[str, Any] = {"solution": [lambda_value], "residual": [f]}

    for it in range(1, max_iterations + 1):
        if abs(f) <= tolerance_f:
            return SolverResult("newton", lambda_value, abs(f), it, True, time.perf_counter() - start, history=history)

        df_dlambda = None
        if not use_numeric_derivative and (dmsign is not None):
            Z = G + torch.tensor(lambda_value, device=G.device, dtype=torch.float32) * Theta
            # Analytic derivative via VJP of msign:
            # df/dλ = <Θ, dΦ/dA[Θ]> with A = G + λΘ
            dA = dmsign(Z, Theta, msign_fn=msign)
            df_dlambda = float(inner_product(dA, Theta).item())

        # FD fallback
        if df_dlambda is None or not math.isfinite(df_dlambda) or df_dlambda == 0.0:
            h = finite_diff_eps * max(1.0, abs(lambda_value))
            f_plus  = compute_f(G, Theta, lambda_value + h, msign_steps)
            f_minus = compute_f(G, Theta, lambda_value - h, msign_steps)
            df_dlambda = (f_plus - f_minus) / (2.0 * h)
            use_numeric_derivative = True

        if df_dlambda == 0.0 or not math.isfinite(df_dlambda):
            break

        step = f / df_dlambda
        next_lambda = lambda_value - step


        lambda_value = next_lambda
        f = compute_f(G, Theta, lambda_value, msign_steps)
        history["solution"].append(lambda_value)
        history["residual"].append(f)

    return SolverResult("newton", lambda_value, abs(f), max_iterations, False, time.perf_counter() - start, history=history)
