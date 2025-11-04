# solver/bisection.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Dict, Any

import torch
from .base import SolverResult, evaluate_objective_and_stats


# -----------------------------------------------------------------------------
# Pure bisection method for monotone f(λ) = <Θ, Φ(λ)> with Φ(λ)=msign(G+λΘ)
# Assumes we already have a bracket [a,b] with f(a) ≤ 0 ≤ f(b).
# Function values are evaluated on GPU; scalar control on CPU.
# -----------------------------------------------------------------------------
@torch.no_grad()
def solve_with_bisection(
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

    # Early exits when an endpoint is already a root
    if fa == 0.0:
        return SolverResult("bisection", a, 0.0, 0, True, 0.0, f_evals, (a, b))
    if fb == 0.0:
        return SolverResult("bisection", b, 0.0, 0, True, 0.0, f_evals, (a, b))

    # If the bracket is invalid, return the better endpoint without iterating
    if fa > 0.0 and fb > 0.0:
        x = a if abs(fa) <= abs(fb) else b
        return SolverResult("bisection", x, min(abs(fa), abs(fb)), 0, False, 0.0, f_evals, (a, b))
    if fa < 0.0 and fb < 0.0:
        x = a if abs(fa) <= abs(fb) else b
        return SolverResult("bisection", x, min(abs(fa), abs(fb)), 0, False, 0.0, f_evals, (a, b))

    # Ensure invariant: fa ≤ 0 ≤ fb (swap if needed when signs are flipped)
    if not (fa <= 0.0 <= fb):
        if fb <= 0.0 <= fa:
            a, b, fa, fb = b, a, fb, fa

    history: Dict[str, Any] = {"solution": [], "residual": []}

    x = 0.5 * (a + b)
    fx, *_ = evaluate_objective_and_stats(G, Theta, x, msign_steps)
    f_evals += 1
    history["solution"].append(x)
    history["residual"].append(fx)

    for it in range(1, max_iterations + 1):
        # Termination: function tolerance or interval tolerance
        # Interval tolerance scales with |x| to be consistent with others
        if abs(fx) <= tolerance_f:
            return SolverResult("bisection", x, abs(fx), it, True, time.perf_counter() - start, f_evals, (a, b), history)

        half_width = 0.5 * (b - a)
        if abs(half_width) <= tolerance_x * max(1.0, abs(x)):
            return SolverResult("bisection", x, abs(fx), it, True, time.perf_counter() - start, f_evals, (a, b), history)

        # Decide which subinterval keeps the root, maintaining fa ≤ 0 ≤ fb
        if fx < 0.0:
            a, fa = x, fx
        else:
            b, fb = x, fx

        x = 0.5 * (a + b)
        fx, *_ = evaluate_objective_and_stats(G, Theta, x, msign_steps)
        f_evals += 1
        history["solution"].append(x)
        history["residual"].append(fx)

    # Max iterations reached
    return SolverResult("bisection", x, abs(fx), max_iterations, False, time.perf_counter() - start, f_evals, (a, b), history)
