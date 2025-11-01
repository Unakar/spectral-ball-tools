# solver/brent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Tuple

import torch
from .base import SolverResult, evaluate_objective_and_stats


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
) -> Tuple[float, float, float, float, int]:
    f_evals = 0
    fa, *_ = evaluate_objective_and_stats(G, Theta, initial_guess, msign_steps)
    f_evals += 1
    if fa == 0.0:
        return initial_guess, initial_guess, fa, fa, f_evals

    step = initial_step if initial_step > 0 else 1.0
    a = b = initial_guess
    fb = fa

    for _ in range(max_expansions):
        # try right
        b = initial_guess + step
        fb, *_ = evaluate_objective_and_stats(G, Theta, b, msign_steps)
        f_evals += 1
        if fa * fb <= 0:
            return (a, b, fa, fb, f_evals) if a <= b else (b, a, fb, fa, f_evals)

        # try left
        a = initial_guess - step
        fa, *_ = evaluate_objective_and_stats(G, Theta, a, msign_steps)
        f_evals += 1
        if fa * fb <= 0:
            return (a, b, fa, fb, f_evals) if a <= b else (b, a, fb, fa, f_evals)

        step *= 2.0

    return min(a, b), max(a, b), fa, fb, f_evals


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
    f_evals = 0

    if fa == 0.0:
        return SolverResult("brent", a, 0.0, 0, True, 0.0, f_evals, (a, b))
    if fb == 0.0:
        return SolverResult("brent", b, 0.0, 0, True, 0.0, f_evals, (a, b))

    c, fc = a, fa
    d = e = b - a
    x, fx = b, fb

    for it in range(1, max_iterations + 1):
        if fx == 0.0:
            return SolverResult("brent", x, 0.0, it, True, time.perf_counter() - start, f_evals, (a, b))
        if fa * fb > 0:
            a, fa = c, fc
            d = e = b - a
        if abs(fa) < abs(fb):
            a, fa, b, fb, c, fc = c, fc, a, fa, b, fb

        tol = 2.0 * tolerance_x * max(1.0, abs(b))
        m = 0.5 * (c - b)
        if abs(m) <= tol or abs(fb) <= tolerance_f:
            return SolverResult("brent", b, abs(fb), it, True, time.perf_counter() - start, f_evals, (a, b))

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

        fb, *_ = evaluate_objective_and_stats(G, Theta, b, msign_steps)
        f_evals += 1

    return SolverResult("brent", b, abs(fb), max_iterations, False, time.perf_counter() - start, f_evals, (a, b))
