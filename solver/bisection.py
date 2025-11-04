# solver/bisection.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Dict, Any

import torch
from .base import SolverResult, compute_f


"""
纯二分法：求解 f(λ)=⟨Θ, Φ(λ)⟩=0，Φ(λ)=msign(G+λΘ)。

前提：
  - 由凸性（∥G+λΘ∥_* 关于 λ 的凸）可知 f(λ) 单调非减；
  - 需提供有效括号区间 [a,b] 使 f(a)≤0≤f(b)。

收敛判据（仅函数值）：
  - 仅当 |f(λ)| ≤ tolerance_f 判定收敛；否则达到迭代上限视为未收敛。

统计：仅统计迭代步数 iterations；不统计函数评估次数。
"""
@torch.no_grad()
def solve_with_bisection(
    G: torch.Tensor,
    Theta: torch.Tensor,
    a: float,
    b: float,
    fa: float,
    fb: float,
    tolerance_f: float = 1e-8,
    max_iterations: int = 100,
    msign_steps: int = 5,
) -> SolverResult:
    start = time.perf_counter()

    # Early exits when an endpoint is already a root
    if fa == 0.0:
        return SolverResult("bisection", a, 0.0, 0, True, 0.0, (a, b))
    if fb == 0.0:
        return SolverResult("bisection", b, 0.0, 0, True, 0.0, (a, b))

    # If the bracket is invalid, return the better endpoint without iterating
    if fa > 0.0 and fb > 0.0:
        x = a if abs(fa) <= abs(fb) else b
        return SolverResult("bisection", x, min(abs(fa), abs(fb)), 0, False, 0.0, (a, b))
    if fa < 0.0 and fb < 0.0:
        x = a if abs(fa) <= abs(fb) else b
        return SolverResult("bisection", x, min(abs(fa), abs(fb)), 0, False, 0.0, (a, b))

    # Ensure invariant: fa ≤ 0 ≤ fb (swap if needed when signs are flipped)
    if not (fa <= 0.0 <= fb):
        if fb <= 0.0 <= fa:
            a, b, fa, fb = b, a, fb, fa

    history: Dict[str, Any] = {"solution": [], "residual": []}

    x = 0.5 * (a + b)
    fx = compute_f(G, Theta, x, msign_steps)
    history["solution"].append(x)
    history["residual"].append(fx)

    for it in range(1, max_iterations + 1):
        # Termination: function tolerance or interval tolerance
        # Interval tolerance scales with |x| to be consistent with others
        if abs(fx) <= tolerance_f:
            return SolverResult("bisection", x, abs(fx), it, True, time.perf_counter() - start, (a, b), history)


        # Decide which subinterval keeps the root, maintaining fa ≤ 0 ≤ fb
        if fx < 0.0:
            a, fa = x, fx
        else:
            b, fb = x, fx

        x = 0.5 * (a + b)
        fx = compute_f(G, Theta, x, msign_steps)
        history["solution"].append(x)
        history["residual"].append(fx)

    # Max iterations reached
    return SolverResult("bisection", x, abs(fx), max_iterations, False, time.perf_counter() - start, (a, b), history)
