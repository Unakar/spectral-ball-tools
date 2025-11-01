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
    evaluate_objective_and_stats,
)

# -----------------------------------------------------------------------------
# Newton's method:
#   Prefer analytic VJP/Jacobian if available; otherwise fall back to FD.
#   We solve f(λ)=<Θ, Φ(λ)>=0 with Φ(λ)=msign(G+λΘ).
# -----------------------------------------------------------------------------
@torch.no_grad()
def solve_with_newton(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    tolerance_f: float = 1e-10,
    tolerance_x: float = 1e-10,
    max_iterations: int = 50,
    msign_steps: int = 5,
    use_numeric_derivative: bool = False,
    finite_diff_eps: float = 1e-4,
) -> SolverResult:
    start = time.perf_counter()
    lambda_value = initial_guess
    f, Phi = evaluate_objective_and_stats(G, Theta, lambda_value, msign_steps)[:2]
    f_evals = 1

    history: Dict[str, Any] = {"solution": [lambda_value], "residual": [f]}

    for it in range(1, max_iterations + 1):
        if abs(f) <= tolerance_f:
            return SolverResult("newton", lambda_value, abs(f), it, True, time.perf_counter() - start, f_evals, history=history)

        df_dlambda = None
        if not use_numeric_derivative and (dmsign_vjp is not None or dmsign is not None):
            Z = G + torch.tensor(lambda_value, device=G.device, dtype=torch.float32) * Theta
            if dmsign_vjp is not None:
                # VJP path: d<f(λ)>/dλ = <Θ, dΦ/dA[Θ]>
                dA = dmsign_vjp(Z, Theta, msign_fn=msign)
                df_dlambda = float(inner_product(dA, Theta).item())
            elif dmsign is not None:
                dPhi = dmsign(Z, Theta, msign_fn=msign)
                df_dlambda = float(inner_product(Theta, dPhi).item())

        # FD fallback
        if df_dlambda is None or not math.isfinite(df_dlambda) or df_dlambda == 0.0:
            h = finite_diff_eps * max(1.0, abs(lambda_value))
            f_plus, *_  = evaluate_objective_and_stats(G, Theta, lambda_value + h, msign_steps)
            f_minus, *_ = evaluate_objective_and_stats(G, Theta, lambda_value - h, msign_steps)
            f_evals += 2
            df_dlambda = (f_plus - f_minus) / (2.0 * h)
            use_numeric_derivative = True

        if df_dlambda == 0.0 or not math.isfinite(df_dlambda):
            break

        step = f / df_dlambda
        next_lambda = lambda_value - step

        if abs(next_lambda - lambda_value) <= tolerance_x * max(1.0, abs(lambda_value)):
            lambda_value = next_lambda
            f, *_ = evaluate_objective_and_stats(G, Theta, lambda_value, msign_steps)
            f_evals += 1
            history["solution"].append(lambda_value)
            history["residual"].append(f)
            return SolverResult("newton", lambda_value, abs(f), it, True, time.perf_counter() - start, f_evals, history=history)

        lambda_value = next_lambda
        f, *_ = evaluate_objective_and_stats(G, Theta, lambda_value, msign_steps)
        f_evals += 1
        history["solution"].append(lambda_value)
        history["residual"].append(f)

    return SolverResult("newton", lambda_value, abs(f), max_iterations, False, time.perf_counter() - start, f_evals, history=history)
