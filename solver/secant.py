# solver/secant.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from typing import Dict, Any

import torch
from .base import SolverResult, evaluate_objective_and_stats


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
    f0, *_ = evaluate_objective_and_stats(G, Theta, x0, msign_steps)
    f1, *_ = evaluate_objective_and_stats(G, Theta, x1, msign_steps)
    f_evals = 2

    history: Dict[str, Any] = {"solution": [x0, x1], "residual": [f0, f1]}

    for it in range(2, max_iterations + 1):
        if abs(f1) <= tolerance_f:
            return SolverResult("secant", x1, abs(f1), it, True, time.perf_counter() - start, f_evals, history=history)

        denom = f1 - f0
        if denom == 0.0 or not math.isfinite(denom):
            break

        x2 = x1 - f1 * (x1 - x0) / denom
        if abs(x2 - x1) <= tolerance_x * max(1.0, abs(x1)):
            x1 = x2
            f1, *_ = evaluate_objective_and_stats(G, Theta, x1, msign_steps)
            f_evals += 1
            history["solution"].append(x1)
            history["residual"].append(f1)
            return SolverResult("secant", x1, abs(f1), it, True, time.perf_counter() - start, f_evals, history=history)

        x0, f0, x1 = x1, f1, x2
        f1, *_ = evaluate_objective_and_stats(G, Theta, x1, msign_steps)
        f_evals += 1
        history["solution"].append(x1)
        history["residual"].append(f1)

    return SolverResult("secant", x1, abs(f1), max_iterations, False, time.perf_counter() - start, f_evals, history=history)
