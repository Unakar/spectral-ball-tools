# solver/brent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Tuple

import torch
from .base import SolverResult, compute_f


"""
Brent 法：求解 f(λ)=⟨Θ, Φ(λ)⟩=0，其中 Φ(λ)=msign(G+λΘ)。

前提与流程：
  - 需已找到括号区间 [a,b] 使 f(a)·f(b)≤0（见 find_bracket）。
  - 迭代融合二分、割线、逆二次插值，保持稳健与效率。

收敛判据（仅函数值）：
  - 仅当 |f(λ)| ≤ tolerance_f 判定收敛；否则达到迭代上限视为未收敛。

统计：只统计迭代步 iterations；不再统计函数评估次数。
"""

# -----------------------------------------------------------------------------
# Bracketing: find [a, b] such that f(a) * f(b) <= 0
# -----------------------------------------------------------------------------
@torch.no_grad()
def find_bracket(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1.0,
    max_expansions: int = 60,
    msign_steps: int = 5,
) -> Tuple[float, float, float, float]:
    fa = compute_f(G, Theta, initial_guess, msign_steps)
    if fa == 0.0:
        return initial_guess, initial_guess, fa, fa

    step = initial_step if initial_step > 0 else 1.0
    a = b = initial_guess
    fb = fa

    for _ in range(max_expansions):
        # try right
        b = initial_guess + step
        fb = compute_f(G, Theta, b, msign_steps)
        if fa * fb <= 0:
            return (a, b, fa, fb) if a <= b else (b, a, fb, fa)

        # try left
        a = initial_guess - step
        fa = compute_f(G, Theta, a, msign_steps)
        if fa * fb <= 0:
            return (a, b, fa, fb) if a <= b else (b, a, fb, fa)

        step *= 2.0

    return min(a, b), max(a, b), fa, fb


# -----------------------------------------------------------------------------
# Brent root-finding; function values are evaluated on GPU
# -----------------------------------------------------------------------------
@torch.no_grad()
def solve_with_brent(
    G: torch.Tensor,
    Theta: torch.Tensor,
    a: float,
    b: float,
    fa: float,
    fb: float,
    tolerance_f: float = 1e-8,
    tolerance_x: float = 1e-10,
    max_iterations: int = 100,
    msign_steps: int = 5,
) -> SolverResult:
    start = time.perf_counter()

    if fa == 0.0:
        return SolverResult("brent", a, 0.0, 0, True, 0.0, (a, b))
    if fb == 0.0:
        return SolverResult("brent", b, 0.0, 0, True, 0.0, (a, b))

    c, fc = a, fa
    d = e = b - a
    x, fx = b, fb

    for it in range(1, max_iterations + 1):
        if fx == 0.0:
            return SolverResult("brent", x, 0.0, it, True, time.perf_counter() - start, (a, b))
        if fa * fb > 0:
            a, fa = c, fc
            d = e = b - a
        if abs(fa) < abs(fb):
            a, fa, b, fb, c, fc = c, fc, a, fa, b, fb

        tol = 2.0 * tolerance_x * max(1.0, abs(b))
        m = 0.5 * (c - b)

        if abs(fb) <= tolerance_f:
            return SolverResult("brent", b, abs(fb), it, True, time.perf_counter() - start, (a, b))

        if abs(e) >= tol and abs(fc) > abs(fb):
            s = fb / fc
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fc / fa
                r = fb / fa
                p = s * (2.0 * m * q * (q - r) - (b - c) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        else:
            d = e = m

        c, fc = b, fb
        if abs(d) > tol:
            b += d
        else:
            b += tol if m > 0 else -tol

        fb = compute_f(G, Theta, b, msign_steps)

    return SolverResult("brent", b, abs(fb), max_iterations, False, time.perf_counter() - start, (a, b))
