import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import torch

from msign import msign
from dmsign import dmsign, dmsign_vjp


def frobenius_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product: sum(x * y)."""
    return (x * y).sum()


def nuclear_norm(a: torch.Tensor) -> torch.Tensor:
    """Sum of singular values."""
    s = torch.linalg.svdvals(a)
    return s.sum()


def polar_factor_via_svd(a: torch.Tensor) -> torch.Tensor:
    """Compute the polar factor U V^T via SVD: A = U S V^T => Polar = U V^T."""
    U, _, Vh = torch.linalg.svd(a, full_matrices=False)
    return U @ Vh


def objective_value(G: torch.Tensor, Theta: torch.Tensor, lam: float, msign_steps: int = 10) -> Tuple[float, torch.Tensor]:
    """Compute f(lam) = <Theta, msign(G + lam * Theta)> and the polar estimate."""
    A = G + lam * Theta
    Phi = msign(A, steps=msign_steps)
    f_val = frobenius_inner(Theta, Phi).item()
    return f_val, Phi


def compute_step_size(A: torch.Tensor, Theta: torch.Tensor) -> float:
    """
    Compute c = ||A||_* / (m * ||Theta||_F^2), used in fixed-point iteration.
    This corresponds to Eq. (10) in the paper.
    """
    m = A.shape[-2]
    tr_Theta_sq = float((Theta * Theta).sum().item())
    trP = float(nuclear_norm(A).item())
    denom = max(1e-12, m * tr_Theta_sq)
    return trP / denom


@dataclass
class SolverResult:
    method: str
    solution: float
    residual: float
    iterations: int
    converged: bool
    time_sec: float = 0.0
    function_evaluations: int = 0
    bracket: Tuple[float, float] | None = None
    history: Dict[str, Any] | None = None


def find_bracket(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    step_size: float = 1.0,
    max_expansions: int = 60,
    msign_steps: int = 10,
) -> Tuple[float, float, float, float, int]:
    """Expand around initial_guess to find an interval [a, b] where f(a)*f(b) <= 0."""
    f_evals = 0
    f0, _ = objective_value(G, Theta, initial_guess, msign_steps)
    f_evals += 1
    if f0 == 0.0:
        return initial_guess, initial_guess, f0, f0, f_evals

    a = b = initial_guess
    fa = fb = f0
    step = step_size if step_size > 0 else 1.0

    for _ in range(max_expansions):
        # Try right
        b = initial_guess + step
        fb, _ = objective_value(G, Theta, b, msign_steps)
        f_evals += 1
        if fa * fb <= 0:
            return (a, b, fa, fb, f_evals) if a <= b else (b, a, fb, fa, f_evals)

        # Try left
        a = initial_guess - step
        fa, _ = objective_value(G, Theta, a, msign_steps)
        f_evals += 1
        if fa * fb <= 0:
            return (a, b, fa, fb, f_evals) if a <= b else (b, a, fb, fa, f_evals)

        step *= 2.0

    return min(a, b), max(a, b), fa, fb, f_evals


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
    msign_steps: int = 10,
) -> SolverResult:
    import time

    if fa == 0.0:
        return SolverResult("brent", a, 0.0, 0, True, 0.0, 0, (a, b))
    if fb == 0.0:
        return SolverResult("brent", b, 0.0, 0, True, 0.0, 0, (a, b))

    c, fc = a, fa
    d = e = b - a
    lam, fl = b, fb
    start_time = time.perf_counter()
    f_evals = 0

    for it in range(1, max_iterations + 1):
        if fl == 0.0:
            return SolverResult("brent", lam, 0.0, it, True, time.perf_counter() - start_time, f_evals, (a, b))
        if fa * fb > 0:
            a, fa = c, fc
            d = e = b - a
        if abs(fa) < abs(fb):
            a, fa, b, fb, c, fc = c, fc, a, fa, b, fb

        tol = 2.0 * tolerance_x * max(1.0, abs(b))
        m = 0.5 * (c - b)
        if abs(m) <= tol or abs(fb) <= tolerance_f:
            return SolverResult("brent", b, abs(fb), it, True, time.perf_counter() - start_time, f_evals, (a, b))

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

        fb, _ = objective_value(G, Theta, b, msign_steps)
        f_evals += 1

    return SolverResult("brent", b, abs(fb), max_iterations, False, time.perf_counter() - start_time, f_evals, (a, b))


def solve_with_secant(
    G: torch.Tensor,
    Theta: torch.Tensor,
    x0: float,
    x1: float,
    tolerance_f: float = 1e-8,
    tolerance_x: float = 1e-10,
    max_iterations: int = 100,
    msign_steps: int = 10,
) -> SolverResult:
    import time

    f0, _ = objective_value(G, Theta, x0, msign_steps)
    f1, _ = objective_value(G, Theta, x1, msign_steps)
    history = {"solution": [x0, x1], "residual": [f0, f1]}
    f_evals = 2
    start_time = time.perf_counter()

    for it in range(2, max_iterations + 1):
        if abs(f1) <= tolerance_f:
            return SolverResult("secant", x1, abs(f1), it, True, time.perf_counter() - start_time, f_evals, history=history)
        denom = f1 - f0
        if denom == 0:
            break
        x2 = x1 - f1 * (x1 - x0) / denom
        if abs(x2 - x1) <= tolerance_x * max(1.0, abs(x1)):
            x1 = x2
            f1, _ = objective_value(G, Theta, x1, msign_steps)
            f_evals += 1
            return SolverResult("secant", x1, abs(f1), it, True, time.perf_counter() - start_time, f_evals, history=history)
        x0, f0, x1 = x1, f1, x2
        f1, _ = objective_value(G, Theta, x1, msign_steps)
        history["solution"].append(x1)
        history["residual"].append(f1)
        f_evals += 1

    return SolverResult("secant", x1, abs(f1), max_iterations, False, time.perf_counter() - start_time, f_evals, history=history)


def solve_with_fixed_point(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    tolerance_f: float = 1e-8,
    max_iterations: int = 200,
    msign_steps: int = 10,
    damping: float = 1.0,
    use_backtracking: bool = True,
) -> SolverResult:
    import time

    lam = initial_guess
    f, _ = objective_value(G, Theta, lam, msign_steps)
    history = {"solution": [lam], "residual": [f]}
    f_evals = 1
    start_time = time.perf_counter()

    for it in range(1, max_iterations + 1):
        if abs(f) <= tolerance_f:
            return SolverResult("fixed_point", lam, abs(f), it, True, time.perf_counter() - start_time, f_evals, history=history)

        A = G + lam * Theta
        step_size = compute_step_size(A, Theta) * max(1e-3, damping)
        new_lam = lam - step_size * f

        new_f, _ = objective_value(G, Theta, new_lam, msign_steps)
        f_evals += 1

        if use_backtracking and abs(new_f) > abs(f):
            success = False
            for _ in range(8):
                step_size *= 0.5
                new_lam = lam - step_size * f
                new_f, _ = objective_value(G, Theta, new_lam, msign_steps)
                f_evals += 1
                if abs(new_f) <= abs(f):
                    success = True
                    break
            if not success:
                pass  # accept anyway

        lam, f = new_lam, new_f
        history["solution"].append(lam)
        history["residual"].append(f)

    return SolverResult("fixed_point", lam, abs(f), max_iterations, False, time.perf_counter() - start_time, f_evals, history=history)


def solve_with_newton(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    tolerance_f: float = 1e-10,
    tolerance_x: float = 1e-10,
    max_iterations: int = 50,
    msign_steps: int = 10,
    use_numeric_derivative: bool = True,
    finite_diff_eps: float = 1e-4,
) -> SolverResult:
    import time

    lam = initial_guess
    f, Phi = objective_value(G, Theta, lam, msign_steps)
    history = {"solution": [lam], "residual": [f]}
    f_evals = 1
    start_time = time.perf_counter()

    for it in range(1, max_iterations + 1):
        if abs(f) <= tolerance_f:
            return SolverResult("newton", lam, abs(f), it, True, time.perf_counter() - start_time, f_evals, history=history)

        if not use_numeric_derivative:
            A = G + lam * Theta
            if dmsign_vjp is not None:
                dA = dmsign_vjp(A, Theta, msign_fn=msign)
                fp = float(frobenius_inner(dA, Theta).item())
            elif dmsign is not None:
                dPhi = dmsign(A, Theta, msign_fn=msign)
                fp = float(frobenius_inner(Theta, dPhi).item())
            else:
                use_numeric_derivative = True

        if use_numeric_derivative:
            h = finite_diff_eps * max(1.0, abs(lam))
            fph, _ = objective_value(G, Theta, lam + h, msign_steps)
            fmh, _ = objective_value(G, Theta, lam - h, msign_steps)
            f_evals += 2
            fp = (fph - fmh) / (2.0 * h)

        if fp == 0.0 or not math.isfinite(fp):
            break

        step = f / fp
        new_lam = lam - step
        if abs(new_lam - lam) <= tolerance_x * max(1.0, abs(lam)):
            lam = new_lam
            f, _ = objective_value(G, Theta, lam, msign_steps)
            f_evals += 1
            history["solution"].append(lam)
            history["residual"].append(f)
            return SolverResult("newton", lam, abs(f), it, True, time.perf_counter() - start_time, f_evals, history=history)

        lam = new_lam
        f, _ = objective_value(G, Theta, lam, msign_steps)
        f_evals += 1
        history["solution"].append(lam)
        history["residual"].append(f)

    return SolverResult("newton", lam, abs(f), max_iterations, False, time.perf_counter() - start_time, f_evals, history=history)


def generate_test_matrices(m: int, n: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random G and normalized Theta (full-rank by default)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    G = torch.randn(m, n, device=device, dtype=torch.float32)
    Theta = torch.randn(m, n, device=device, dtype=torch.float32)
    Theta = Theta / (Theta.norm() + 1e-12)  # Normalize to avoid scale issues
    return G, Theta


def run_solver_demo(method: str = "brent") -> None:
    """Run a single demo case with clean output."""
    G, Theta = generate_test_matrices(m=64, n=32, seed=42)
    msign_steps = 5
    tol = 1e-8
    max_iter = 100

    print("=== Demo: Solving ⟨Θ, msign(G + λΘ)⟩ = 0 ===")

    # Baseline at λ=0
    f0, Phi0 = objective_value(G, Theta, 0.0, msign_steps)
    Phi_ref0 = polar_factor_via_svd(G)
    rel_err0 = (Phi0 - Phi_ref0).norm().item() / (Phi_ref0.norm().item() + 1e-12)
    print(f"f(0) = {f0:.6e}, polar error at λ=0: {rel_err0:.3e}")

    # Solve
    if method == "brent":
        a, b, fa, fb, _ = find_bracket(G, Theta, msign_steps=msign_steps)
        if fa * fb > 0:
            print("[Warning] Could not bracket root; falling back to secant.")
            result = solve_with_secant(G, Theta, a, b, tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
        else:
            result = solve_with_brent(G, Theta, a, b, fa, fb, tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
    elif method == "secant":
        a, b, _, _, _ = find_bracket(G, Theta, msign_steps=msign_steps)
        x0, x1 = (a, b) if a != b else (0.0, 1.0)
        result = solve_with_secant(G, Theta, x0, x1, tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
    elif method == "fixed_point":
        result = solve_with_fixed_point(G, Theta, tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps)
    elif method == "newton":
        result = solve_with_newton(G, Theta, tolerance_f=tol, max_iterations=max_iter, msign_steps=msign_steps, use_numeric_derivative=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Final check
    f_sol, Phi_sol = objective_value(G, Theta, result.solution, msign_steps)
    Phi_ref_sol = polar_factor_via_svd(G + result.solution * Theta)
    rel_err_sol = (Phi_sol - Phi_ref_sol).norm().item() / (Phi_ref_sol.norm().item() + 1e-12)

    print(f"[{result.method}] λ* = {result.solution:.10f}")
    print(f"  |f(λ*)| = {result.residual:.3e} (target ≤ {tol})")
    print(f"  Iterations: {result.iterations}, Function evals: {result.function_evaluations}")
    print(f"  Converged: {result.converged}, Time: {result.time_sec * 1000:.2f} ms")
    print(f"  Polar error at λ*: {rel_err_sol:.3e}")


if __name__ == "__main__":
    run_solver_demo(method="brent")