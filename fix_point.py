# -*- coding: utf-8 -*-
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
from msign import msign  # Use your high-precision / efficient Polar Express msign implementation


# -----------------------------------------------------------------------------
# Basic utilities (FP32 accumulation to reduce mixed-precision error)
# -----------------------------------------------------------------------------
@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute <a, b> with FP32 accumulation; returns a scalar tensor on GPU."""
    return (a.to(torch.float32) * b.to(torch.float32)).sum()


@torch.no_grad()
def trace_fp32(a: torch.Tensor) -> torch.Tensor:
    """Compute trace(a) with FP32 accumulation; returns a scalar tensor on GPU."""
    return torch.trace(a.to(torch.float32))


# -----------------------------------------------------------------------------
# Top singular vectors via bilateral power iteration (SVD-free, GPU-friendly)
# Iteration:
#   1) u ← normalize(W v)
#   2) v ← normalize(W^T u)
# Rayleigh quotient approximates the largest singular value: sigma ≈ u^T (W v)
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_top_singular_vectors(
    W: torch.Tensor,
    iters: int = 30,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns (u1, v1, sigma_est). All computation is on GPU.
    Note: Normalization is performed in FP32 to reduce scale drift; only GEMV/GEMM ops are used.
    """
    assert W.is_cuda, "This script expects GPU (CUDA)."
    n, m = W.shape

    v = torch.randn(m, device=W.device, dtype=torch.float32)
    v = v / (v.norm() + 1e-20)

    sigma_prev = 0.0
    for _ in range(iters):
        u = W @ v
        u_norm = u.norm()
        if u_norm < 1e-30:
            # Degenerate case: return zero vectors and zero singular value
            u = torch.zeros_like(u)
            v = torch.zeros_like(v)
            return u, v, 0.0
        u = u / u_norm

        v = W.mT @ u
        v_norm = v.norm()
        if v_norm < 1e-30:
            u = torch.zeros_like(u)
            v = torch.zeros_like(v)
            return u, v, 0.0
        v = v / v_norm

        sigma = torch.dot(u, W @ v).item()
        if abs(sigma - sigma_prev) < tol * max(1.0, abs(sigma_prev)):
            break
        sigma_prev = sigma

    return u, v, float(sigma_prev)


# -----------------------------------------------------------------------------
# Fixed-point solver for λ (convergence verification only)
# Key identity: for z = G + λΘ, let Φ = msign(z), then
#   q := z^T Φ  ≡  (z^T z)^{1/2}
# Thus, q can replace Q explicitly, avoiding matrix square root or inverse.
# Update rule (matches the reference derivation):
#   λ ← [ tr(Θ^T Φ q) - tr(Θ^T Φ) tr(q) / m - tr(Θ^T G) ] / tr(Θ^T Θ)
# -----------------------------------------------------------------------------
@dataclass
class LambdaIterLog:
    step: int
    lambda_value: float          # For logging; stored as Python float
    constraint_abs: float        # |tr(Θ^T Φ)|
    objective_value: float       # <G, Φ>
    ms_per_step: float           # Time per iteration (milliseconds)


@torch.no_grad()
def solve_lambda_fixed_point(
    G: torch.Tensor,
    Theta: torch.Tensor,
    tol_abs_constraint: float = 1e-5,
    max_iterations: int = 1000,
    msign_steps: int = 5,
    verbose: bool = False,
) -> Tuple[float, torch.Tensor, List[LambdaIterLog]]:
    """
    Solve for λ such that tr(Θ^T Φ) = 0, where Φ = msign(G + λΘ).

    Design notes:
      - All scalars remain on GPU; conversion to Python float occurs only for logging/printing.
      - Dual convergence criterion: both |tr(Θ^T Φ)| and relative λ change must fall below tolerance.
    """
    assert G.is_cuda and Theta.is_cuda, "Expected CUDA tensors."
    n, m = G.shape

    # Precompute scalar quantities (as GPU tensors)
    trace_theta_t_theta = trace_fp32(Theta.mT @ Theta)      # tr(Θ^T Θ)
    trace_theta_t_g     = trace_fp32(Theta.mT @ G)          # tr(Θ^T G)

    # Initialize λ = - tr(Θ^T G) / tr(Θ^T Θ)
    lambda_value = -trace_theta_t_g / (trace_theta_t_theta + 1e-30)

    logs: List[LambdaIterLog] = []
    prev_lambda_value = None

    for k in range(max_iterations):
        # Time using CUDA events to avoid implicit synchronization
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_stop  = torch.cuda.Event(enable_timing=True)
        cuda_start.record()

        z   = G + lambda_value * Theta
        Phi = msign(z, steps=msign_steps)
        X   = Theta.mT @ Phi               # X = Θ^T Φ
        q   = z.mT @ Phi                   # q = z^T Φ = (z^T z)^{1/2}

        tr_X    = trace_fp32(X)
        tr_q    = trace_fp32(q)
        tr_Xq   = trace_fp32(X @ q)
        constraint_abs = tr_X.abs()

        # Relative λ change (as GPU scalar tensor)
        if prev_lambda_value is None:
            rel_lambda_change = torch.tensor(float("inf"), device=G.device)
        else:
            rel_lambda_change = (lambda_value - prev_lambda_value).abs() / (lambda_value.abs() + 1e-12)

        cuda_stop.record()
        torch.cuda.synchronize()
        step_ms = cuda_start.elapsed_time(cuda_stop)

        # Log (convert to Python floats here)
        logs.append(
            LambdaIterLog(
                step=k,
                lambda_value=float(lambda_value.item()),
                constraint_abs=float(constraint_abs.item()),
                objective_value=float(inner_product(G, Phi).item()),
                ms_per_step=float(step_ms),
            )
        )

        if verbose and (k % 50 == 0 or (constraint_abs < tol_abs_constraint and rel_lambda_change < tol_abs_constraint)):
            print(
                f"[λ {k:4d}] |tr(ΘᵀΦ)|={float(constraint_abs.item()):.3e}  "
                f"Δλ/|λ|={float(rel_lambda_change.item()):.3e}  "
                f"λ={float(lambda_value.item()):.6e}  {step_ms:.2f}ms"
            )

        # Dual convergence: both constraint residual and relative λ change below threshold
        if constraint_abs < tol_abs_constraint and rel_lambda_change < tol_abs_constraint:
            return float(lambda_value.item()), Phi, logs

        # Fixed-point update (on GPU)
        numerator = tr_Xq - tr_X * tr_q / m - trace_theta_t_g
        prev_lambda_value = lambda_value
        lambda_value = numerator / (trace_theta_t_theta + 1e-30)

    return float(lambda_value.item()), Phi, logs


# -----------------------------------------------------------------------------
# Main entry point:
#   - Randomly initialize W and G
#   - Construct Θ = u1 v1^T using top singular vectors of W (via power iteration)
#   - Verify convergence of λ fixed-point iteration
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of rows (n >= m)")
    parser.add_argument("--m", type=int, default=50,  help="Number of columns")
    parser.add_argument("--tol", type=float, default=1e-5, help="Convergence tolerance for |tr(Θ^T Φ)| and relative λ change")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations for λ fixed-point solver")
    parser.add_argument("--msign_steps", type=int, default=5, help="Number of steps in msign (Polar Express)")
    parser.add_argument("--power_iters", type=int, default=30, help="Number of power iterations to compute u1, v1")
    parser.add_argument("--verbose", action="store_true", help="Print iteration logs")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available; this script is GPU-only."
    device = torch.device("cuda")

    # Randomly initialize W and G (used to construct Θ and as optimization target)
    W = torch.randn(args.n, args.m, device=device, dtype=torch.float32) / math.sqrt(args.m)
    G = torch.randn(args.n, args.m, device=device, dtype=torch.float32) / math.sqrt(args.m)

    # Θ = u1 v1^T, where (u1, v1) are computed via bilateral power iteration (no SVD)
    u1, v1, _ = compute_top_singular_vectors(W, iters=args.power_iters, tol=1e-6)
    Theta = torch.outer(u1, v1)  # shape: n x m

    # Solve for λ and verify convergence
    lambda_value, Phi, history = solve_lambda_fixed_point(
        G=G,
        Theta=Theta,
        tol_abs_constraint=args.tol,
        max_iterations=args.max_iter,
        msign_steps=args.msign_steps,
        verbose=args.verbose,
    )

    last = history[-1]
    print("\n=== Summary ===")
    print(f"size: {args.n}x{args.m}")
    print(f"iters: {len(history)}")
    print(f"final |tr(ΘᵀΦ)|: {last.constraint_abs:.3e}")
    print(f"final λ: {lambda_value:.9e}")
    print(f"final obj <G,Φ>: {last.objective_value:.6f}")
    print(f"avg ms/iter: {sum(h.ms_per_step for h in history)/len(history):.2f} ms")


if __name__ == "__main__":
    main()
