# solver/secant.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Dict, Any

import torch
from .base import SolverResult, compute_f


"""
割线法：求解 f(λ)=⟨Θ, Φ(λ)⟩=0，Φ(λ)=msign(G+λΘ)。

更新：
  λ_{k+1} = λ_k − f(λ_k)·(λ_k−λ_{k−1}) / (f(λ_k)−f(λ_{k−1}))。

收敛判据（仅函数值）：
  - 仅当 |f(λ)| ≤ tolerance_f 判定收敛；否则达到迭代上限视为未收敛。

统计：仅统计迭代步数 iterations；不统计函数评估次数。
"""

# -----------------------------------------------------------------------------
# Secant method (function-value only; f is evaluated on GPU)
# -----------------------------------------------------------------------------
@torch.no_grad()
def solve_with_secant(
    G: torch.Tensor,
    Theta: torch.Tensor,
    x0: float,
    x1: float,
    tolerance_f: float = 1e-8,
    tolerance_x: float = 1e-10,
    max_iterations: int = 100,
    msign_steps: int = 5,
) -> SolverResult:
    start = time.perf_counter()
    f0 = compute_f(G, Theta, x0, msign_steps)
    f1 = compute_f(G, Theta, x1, msign_steps)

    history: Dict[str, Any] = {"solution": [x0, x1], "residual": [f0, f1]}

    for it in range(2, max_iterations + 1):
        if abs(f1) <= tolerance_f:
            return SolverResult("secant", x1, abs(f1), it, True, time.perf_counter() - start, history=history)

        denom = f1 - f0
        if denom == 0.0 or not math.isfinite(denom):
            break

        x2 = x1 - f1 * (x1 - x0) / denom
        # 不使用 |Δλ| 的早停，继续迭代直到 |f| 达标或达到上限

        x0, f0, x1 = x1, f1, x2
        f1 = compute_f(G, Theta, x1, msign_steps)
        history["solution"].append(x1)
        history["residual"].append(f1)

    return SolverResult("secant", x1, abs(f1), max_iterations, False, time.perf_counter() - start, history=history)
