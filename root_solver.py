# root_solver.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import torch

from solver.base import (
    compute_f,
    compute_phi,
)
from kernels.power_iteration import power_iteration, top_svd
from solver.brent import find_bracket, solve_with_brent
from solver.secant import solve_with_secant
from solver.newton import solve_with_newton
from solver.fix_point import solve_with_fixed_point  
from solver.bisection import solve_with_bisection


@torch.no_grad()
def run_solver_demo(method: str, n: int, m: int, seed: int, tol: float, max_iter: int, msign_steps: int,
                    theta_source: str = "power", power_iters: int = 3) -> None:
    # Random test matrices G and W; construct Θ from W via selected method
    assert torch.cuda.is_available(), "CUDA not available; GPU is required."
    device = torch.device("cuda")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    W = torch.randn(n, m, device=device, dtype=torch.float32) 
    G = torch.randn(n, m, device=device, dtype=torch.float32) 

    if theta_source == "power":
        _, u, v = power_iteration(W, steps=power_iters)
        Theta = u @ v.mT
    elif theta_source == "svd":
        _, u, v = top_svd(W)
        Theta = u @ v.mT
    else:
        raise ValueError(f"Unknown theta_source: {theta_source}")

    print(f"=== Solve  ⟨Θ, msign(G + λΘ)⟩ = 0  ===  \n (n={n}, m={m}, method={method})")

    if method == "brent":
        a, b, fa, fb = find_bracket(G, Theta, msign_steps=msign_steps)
        if fa * fb > 0:
            print("[warn] could not bracket a root; fallback to secant.")
            result = solve_with_secant(G, Theta, a, b,
                                       tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
            result.bracket = (a, b)
        else:
            result = solve_with_brent(G, Theta, a, b, fa, fb,
                                      tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)

    elif method == "bisection":
        a, b, fa, fb = find_bracket(G, Theta, msign_steps=msign_steps)
        if fa * fb > 0:
            print("[warn] could not bracket a root; fallback to secant.")
            result = solve_with_secant(G, Theta, a, b,
                                       tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
            result.bracket = (a, b)
        else:
            result = solve_with_bisection(G, Theta, a, b, fa, fb,
                                          tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)

    elif method == "secant":
        a, b, _, _ = find_bracket(G, Theta, msign_steps=msign_steps)
        x0, x1 = (a, b) if a != b else (0.0, 1.0)
        result = solve_with_secant(G, Theta, x0, x1,
                                   tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
        result.bracket = (a, b)

    elif method == "fixed_point":
        result = solve_with_fixed_point(G, Theta,
                                        tolerance_f=tol, max_iterations=max_iter,
                                        msign_steps=msign_steps)

    elif method == "newton":
        result = solve_with_newton(G, Theta,
                                   tolerance_f=tol, max_iterations=max_iter,
                                   msign_steps=msign_steps,
                                   use_numeric_derivative=False)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Unified epilogue
    f_star = compute_f(G, Theta, result.solution, msign_steps)

    print(f"\n[{result.method}]  λ* = {result.solution:.10f}")
    print(f"  |f(λ*)|     : {abs(f_star):.3e}  (target ≤ {tol})")
    print(f"  iters       : {result.iterations} iters")
    print(f"  converged   : {result.converged}")
    print(f"  time        : {result.time_sec*1000:.2f} ms")
    if result.bracket is not None:
        print(f"  bracket     : [{result.bracket[0]:.4g}, {result.bracket[1]:.4g}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="brent",
                        choices=["brent", "bisection", "secant", "fixed_point", "newton"])
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--msign_steps", type=int, default=5)
    parser.add_argument("--theta_source", type=str, default="power", choices=["power", "svd"],
                        help="How to build Θ=u v^T from W: power (default) or svd")
    parser.add_argument("--power_iters", type=int, default=30,help="Iterations for power method to build Θ")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available; GPU is required."
    run_solver_demo(args.method, args.n, args.m, args.seed, args.tol, args.max_iter, args.msign_steps,
                    theta_source=args.theta_source, power_iters=args.power_iters)


if __name__ == "__main__":
    main()
