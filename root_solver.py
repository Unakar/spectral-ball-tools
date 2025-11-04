# root_solver.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import torch

from solver.base import (
    generate_test_matrices,
    evaluate_objective_and_stats,
)
from solver.brent import find_bracket, solve_with_brent
from solver.secant import solve_with_secant
from solver.newton import solve_with_newton
from solver.fix_point import solve_lambda_fixed_point  
from solver.bisection import solve_with_bisection


@torch.no_grad()
def run_solver_demo(method: str, n: int, m: int, seed: int, tol: float, max_iter: int, msign_steps: int) -> None:
    G, Theta = generate_test_matrices(n, m, seed=seed)

    print(f"=== Solve  ⟨Θ, msign(G + λΘ)⟩ = 0  ===  (n={n}, m={m}, method={method})")
    f0, Phi0, _, _ = evaluate_objective_and_stats(G, Theta, 0.0, msign_steps)
    ortho_err0 = (Phi0.mT @ Phi0 - torch.eye(m, device=G.device)).norm().item() / (m + 1e-12)
    print(f"f(0)={f0:.6e}  |  orthogonality error @λ=0: {ortho_err0:.3e}")

    if method == "brent":
        a, b, fa, fb, f_evals = find_bracket(G, Theta, msign_steps=msign_steps)
        if fa * fb > 0:
            print("[warn] could not bracket a root; fallback to secant.")
            result = solve_with_secant(G, Theta, a, b,
                                       tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
            result.function_evaluations += f_evals
            result.bracket = (a, b)
        else:
            result = solve_with_brent(G, Theta, a, b, fa, fb,
                                      tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
            result.function_evaluations += f_evals

    elif method == "bisection":
        a, b, fa, fb, f_evals = find_bracket(G, Theta, msign_steps=msign_steps)
        if fa * fb > 0:
            print("[warn] could not bracket a root; fallback to secant.")
            result = solve_with_secant(G, Theta, a, b,
                                       tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
            result.function_evaluations += f_evals
            result.bracket = (a, b)
        else:
            result = solve_with_bisection(G, Theta, a, b, fa, fb,
                                          tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
            result.function_evaluations += f_evals

    elif method == "secant":
        a, b, _, _, f_evals = find_bracket(G, Theta, msign_steps=msign_steps)
        x0, x1 = (a, b) if a != b else (0.0, 1.0)
        result = solve_with_secant(G, Theta, x0, x1,
                                   tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
        result.function_evaluations += f_evals
        result.bracket = (a, b)

    elif method == "fixed_point":
        lambda_value, Phi, history = solve_lambda_fixed_point(
            G=G, Theta=Theta, tol_abs_constraint=tol, max_iterations=max_iter,
            msign_steps=msign_steps, verbose=True
        )
        last = history[-1]
        print(f"\n[fixed_point]  λ* = {lambda_value:.10f}")
        print(f"  |tr(ΘᵀΦ)| : {last.constraint_abs:.3e}")
        print(f"  iters     : {len(history)}")
        print(f"  time      : {sum(h.ms_per_step for h in history):.2f} ms (sum)  "
              f"{sum(h.ms_per_step for h in history)/len(history):.2f} ms/iter (avg)")
        f_star, Phi_star, _, _ = evaluate_objective_and_stats(G, Theta, lambda_value, msign_steps)
        ortho_err_star = (Phi_star.mT @ Phi_star - torch.eye(m, device=G.device)).norm().item() / (m + 1e-12)
        print(f"  |f(λ*)|   : {abs(f_star):.3e}")
        print(f"  orthogonality error @λ*: {ortho_err_star:.3e}")
        return

    elif method == "newton":
        result = solve_with_newton(G, Theta,
                                   tolerance_f=tol, max_iterations=max_iter,
                                   msign_steps=msign_steps,
                                   use_numeric_derivative=False)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Unified epilogue
    f_star, Phi_star, _, _ = evaluate_objective_and_stats(G, Theta, result.solution, msign_steps)
    ortho_err_star = (Phi_star.mT @ Phi_star - torch.eye(m, device=G.device)).norm().item() / (m + 1e-12)

    print(f"\n[{result.method}]  λ* = {result.solution:.10f}")
    print(f"  |f(λ*)|     : {abs(f_star):.3e}  (target ≤ {tol})")
    print(f"  iters/evals : {result.iterations} iters, {result.function_evaluations} f-evals")
    print(f"  converged   : {result.converged}")
    print(f"  time        : {result.time_sec*1000:.2f} ms")
    if result.bracket is not None:
        print(f"  bracket     : [{result.bracket[0]:.4g}, {result.bracket[1]:.4g}]")
    print(f"  orthogonality error @λ*: {ortho_err_star:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="brent",
                        choices=["brent", "bisection", "secant", "fixed_point", "newton"])
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--msign_steps", type=int, default=5)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available; GPU is required."
    run_solver_demo(args.method, args.n, args.m, args.seed, args.tol, args.max_iter, args.msign_steps)


if __name__ == "__main__":
    main()
