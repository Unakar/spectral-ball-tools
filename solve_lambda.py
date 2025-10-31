import argparse
import math
from dataclasses import dataclass
import time
from typing import Callable, Optional, Tuple, Dict, Any

import torch
import importlib.util
from pathlib import Path

# Local operators: load directly from file to avoid package requirements
_HERE = Path(__file__).resolve().parent
_MSIGN_PY = _HERE / "msign.py"
_DMSIGN_PY = _HERE / "dmsign.py"


def _load_function(module_path: Path, func_name: str):
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name, None)


phi_msign = _load_function(_MSIGN_PY, "msign")
if phi_msign is None:
    raise ImportError(f"Cannot load msign from {_MSIGN_PY}")

dmsign_op = _load_function(_DMSIGN_PY, "dmsign")
dmsign_vjp_op = _load_function(_DMSIGN_PY, "dmsign_vjp")


Tensor = torch.Tensor


def set_device(dev: str) -> torch.device:
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def frob_inner(x: Tensor, y: Tensor) -> Tensor:
    return (x * y).sum()


def nuclear_norm(a: Tensor) -> Tensor:
    # sum of singular values
    s = torch.linalg.svdvals(a)
    return s.sum()


def phi_ref_polar(a: Tensor) -> Tensor:
    # Reference polar factor via SVD: A = U S V^T => U_polar = U V^T (rectangular-friendly)
    U, _, Vh = torch.linalg.svd(a, full_matrices=False)
    return U @ Vh


def f_value(G: Tensor, Theta: Tensor, lam: float, msign_steps: int = 10) -> Tuple[float, Tensor]:
    A = G + lam * Theta
    Phi = phi_msign(A, steps=msign_steps)
    f = frob_inner(Theta, Phi).item()
    return f, Phi


def eq10_step_c(A: Tensor, Theta: Tensor) -> float:
    # c(lam) = tr(P) / (m * tr(Theta^T Theta)) ; with P = (A^T A)^{1/2}
    # tr(P) = nuclear norm of A
    m_rows = A.shape[-2]
    tr_Theta2 = float((Theta * Theta).sum().item())
    trP = float(nuclear_norm(A).item())
    # Guard against divide-by-zero
    denom = max(1e-12, m_rows * tr_Theta2)
    return trP / denom


@dataclass
class SolveStats:
    method: str
    lam: float
    f_abs: float
    iterations: int
    converged: bool
    time_sec: float = 0.0
    f_evals: int = 0
    bracket: Optional[Tuple[float, float]] = None
    history: Optional[Dict[str, Any]] = None


def bracket_find(
    G: Tensor,
    Theta: Tensor,
    lam0: float = 0.0,
    init_step: float = 1.0,
    max_expand: int = 60,
    msign_steps: int = 10,
) -> Tuple[float, float, float, float, int]:
    f_evals = 0
    f0, _ = f_value(G, Theta, lam0, msign_steps)
    f_evals += 1
    if f0 == 0.0:
        return lam0, lam0, f0, f0, f_evals

    # Expand in both directions to find sign change
    a = lam0
    b = lam0
    fa = f0
    fb = f0
    step = init_step if init_step > 0 else 1.0

    for k in range(max_expand):
        # Expand right
        b = lam0 + step
        fb, _ = f_value(G, Theta, b, msign_steps)
        f_evals += 1
        if fa * fb <= 0:
            return ((a, b, fa, fb, f_evals) if a <= b else (b, a, fb, fa, f_evals))

        # Expand left
        a = lam0 - step
        fa, _ = f_value(G, Theta, a, msign_steps)
        f_evals += 1
        if fa * fb <= 0:
            return ((a, b, fa, fb, f_evals) if a <= b else (b, a, fb, fa, f_evals))

        step *= 2.0

    # As a fallback, return the widest interval checked
    return (min(a, b), max(a, b), fa, fb, f_evals)


def brent_solve(
    G: Tensor,
    Theta: Tensor,
    a: float,
    b: float,
    fa: float,
    fb: float,
    tol_f: float = 1e-8,
    tol_x: float = 1e-10,
    max_iter: int = 100,
    msign_steps: int = 10,
) -> SolveStats:
    # Based on Brent-Dekker method
    if fa == 0.0:
        return SolveStats("brent", a, 0.0, 0, True, 0.0, 0, (a, b))
    if fb == 0.0:
        return SolveStats("brent", b, 0.0, 0, True, 0.0, 0, (a, b))
    if fa * fb > 0:
        # Not bracketed; degrade to bisection on [a, b]
        pass

    c = a
    fc = fa
    d = e = b - a
    lam = b
    fl = fb
    iterations = 0

    t0 = time.perf_counter()
    f_evals = 0
    for iterations in range(1, max_iter + 1):
        if fl == 0.0:
            return SolveStats("brent", lam, 0.0, iterations, True, time.perf_counter()-t0, f_evals, (a, b))
        if fa * fb > 0:
            a, fa = c, fc
            d = e = b - a
        if abs(fa) < abs(fb):
            c, fc, a, fa, b, fb = b, fb, a, fa, c, fc

        # Convergence tests
        tol = 2.0 * tol_x * max(1.0, abs(b))
        m = 0.5 * (a - b)
        if abs(m) <= tol or abs(fb) <= tol_f:
            return SolveStats("brent", b, abs(fb), iterations, True, time.perf_counter()-t0, f_evals, (a, b))

        # Attempt inverse quadratic interpolation
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
            accept = (2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)))
            if accept:
                e, d = d, p / q
            else:
                d = m
                e = m
        else:
            d = m
            e = m

        # Move
        c, fc = b, fb
        if abs(d) > tol:
            b += d
        else:
            b += tol if m > 0 else -tol

        fb, _ = f_value(G, Theta, b, msign_steps)
        f_evals += 1

    return SolveStats("brent", b, abs(fb), iterations, False, time.perf_counter()-t0, f_evals, (a, b))


def secant_solve(
    G: Tensor,
    Theta: Tensor,
    lam0: float,
    lam1: float,
    tol_f: float = 1e-8,
    tol_x: float = 1e-10,
    max_iter: int = 100,
    msign_steps: int = 10,
) -> SolveStats:
    t0 = time.perf_counter()
    f0, _ = f_value(G, Theta, lam0, msign_steps)
    f1, _ = f_value(G, Theta, lam1, msign_steps)
    iterations = 0
    history = {"lam": [lam0, lam1], "f": [f0, f1]}
    f_evals = 2

    for iterations in range(2, max_iter + 1):
        if abs(f1) <= tol_f:
            return SolveStats("secant", lam1, abs(f1), iterations, True, time.perf_counter()-t0, f_evals, history=history)
        denom = (f1 - f0)
        if denom == 0:
            break
        lam2 = lam1 - f1 * (lam1 - lam0) / denom
        if abs(lam2 - lam1) <= tol_x * max(1.0, abs(lam1)):
            lam1 = lam2
            f1, _ = f_value(G, Theta, lam1, msign_steps)
            f_evals += 1
            return SolveStats("secant", lam1, abs(f1), iterations, True, time.perf_counter()-t0, f_evals, history=history)
        lam0, f0, lam1 = lam1, f1, lam2
        f1, _ = f_value(G, Theta, lam1, msign_steps)
        history["lam"].append(lam1)
        history["f"].append(f1)
        f_evals += 1

    return SolveStats("secant", lam1, abs(f1), iterations, False, time.perf_counter()-t0, f_evals, history=history)


def fixed_point_solve(
    G: Tensor,
    Theta: Tensor,
    lam0: float = 0.0,
    tol_f: float = 1e-8,
    max_iter: int = 200,
    msign_steps: int = 10,
    damping: float = 1.0,
    backtrack: bool = True,
) -> SolveStats:
    lam = lam0
    f, _ = f_value(G, Theta, lam, msign_steps)
    history = {"lam": [lam], "f": [f]}
    iterations = 0
    t0 = time.perf_counter()
    f_evals = 1

    for iterations in range(1, max_iter + 1):
        if abs(f) <= tol_f:
            return SolveStats("fixed", lam, abs(f), iterations, True, time.perf_counter()-t0, f_evals, history=history)
        A = G + lam * Theta
        c = eq10_step_c(A, Theta) * max(1e-3, damping)
        new_lam = lam - c * f

        new_f, _ = f_value(G, Theta, new_lam, msign_steps)
        f_evals += 1
        if backtrack and abs(new_f) > abs(f):
            # Try half steps to avoid oscillation
            ok = False
            for _ in range(8):
                c *= 0.5
                new_lam = lam - c * f
                new_f, _ = f_value(G, Theta, new_lam, msign_steps)
                f_evals += 1
                if abs(new_f) <= abs(f):
                    ok = True
                    break
            if not ok:
                # accept the best so far even if not decreasing much
                pass

        lam = new_lam
        f = new_f
        history["lam"].append(lam)
        history["f"].append(f)

    return SolveStats("fixed", lam, abs(f), iterations, False, time.perf_counter()-t0, f_evals, history=history)


def newton_solve(
    G: Tensor,
    Theta: Tensor,
    lam0: float = 0.0,
    tol_f: float = 1e-10,
    tol_x: float = 1e-10,
    max_iter: int = 50,
    msign_steps: int = 10,
    numeric_derivative: bool = True,
    fd_eps: float = 1e-4,
) -> SolveStats:
    lam = lam0
    f, Phi = f_value(G, Theta, lam, msign_steps)
    history = {"lam": [lam], "f": [f]}
    iterations = 0
    t0 = time.perf_counter()
    f_evals = 1

    for iterations in range(1, max_iter + 1):
        if abs(f) <= tol_f:
            return SolveStats("newton", lam, abs(f), iterations, True, time.perf_counter()-t0, f_evals, history=history)

        if (dmsign_vjp_op is not None) and (not numeric_derivative):
            A = G + lam * Theta
            dA = dmsign_vjp_op(A, Theta, msign_fn=phi_msign)
            fp = float(frob_inner(dA, Theta).item())
        elif (dmsign_op is not None) and (not numeric_derivative):
            A = G + lam * Theta
            dPhi = dmsign_op(A, Theta, msign_fn=phi_msign)
            fp = float(frob_inner(Theta, dPhi).item())
        else:
            # Finite-difference derivative fallback
            h = fd_eps * max(1.0, abs(lam))
            fph, _ = f_value(G, Theta, lam + h, msign_steps)
            fmh, _ = f_value(G, Theta, lam - h, msign_steps)
            f_evals += 2
            fp = (fph - fmh) / (2.0 * h)

        if fp == 0.0 or not math.isfinite(fp):
            break
        step = f / fp
        new_lam = lam - step
        if abs(new_lam - lam) <= tol_x * max(1.0, abs(lam)):
            lam = new_lam
            f, _ = f_value(G, Theta, lam, msign_steps)
            f_evals += 1
            history["lam"].append(lam)
            history["f"].append(f)
            return SolveStats("newton", lam, abs(f), iterations, True, time.perf_counter()-t0, f_evals, history=history)
        lam = new_lam
        f, _ = f_value(G, Theta, lam, msign_steps)
        f_evals += 1
        history["lam"].append(lam)
        history["f"].append(f)

    return SolveStats("newton", lam, abs(f), iterations, False, time.perf_counter()-t0, f_evals, history=history)


def generate_case(
    m: int,
    n: int,
    rank1_theta: bool = True,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    G = torch.randn(m, n, device=device, dtype=dtype)

    if rank1_theta:
        u = torch.randn(m, 1, device=device, dtype=dtype)
        v = torch.randn(1, n, device=device, dtype=dtype)
        Theta = u @ v  # rank-1
    else:
        Theta = torch.randn(m, n, device=device, dtype=dtype)

    # Normalize Theta to avoid scaling pathologies
    Theta = Theta / (Theta.norm() + 1e-12)
    return G, Theta


def run_one(
    method: str,
    m: int,
    n: int,
    device: torch.device,
    seed: int,
    msign_steps: int,
    tol: float,
    max_iter: int,
    rank1_theta: bool,
    verbose: bool = True,
    compare_ref_phi: bool = True,
) -> None:
    G, Theta = generate_case(m, n, rank1_theta, device=device, dtype=torch.float32, seed=seed)

    # Baseline function at 0
    f0, Phi0 = f_value(G, Theta, 0.0, msign_steps)

    # Optional: compare msign against high-precision polar reference
    if compare_ref_phi:
        with torch.no_grad():
            Phi_ref0 = phi_ref_polar(G)
            rel_phi_err = float((Phi0 - Phi_ref0).norm().item() / (Phi_ref0.norm().item() + 1e-12))
    else:
        rel_phi_err = float("nan")

    if verbose:
        print(f"Case seed={seed} | f(0)={f0:.6e} | rel_phi_err@0={rel_phi_err:.3e}")

    stats: SolveStats
    if method == "brent":
        a, b, fa, fb, _ = bracket_find(G, Theta, 0.0, init_step=1.0, max_expand=60, msign_steps=msign_steps)
        if fa * fb > 0:
            print("[brent] Failed to bracket a sign change. Falling back to secant from endpoints.")
            stats = secant_solve(G, Theta, a, b, tol_f=tol, max_iter=max_iter, msign_steps=msign_steps)
        else:
            stats = brent_solve(G, Theta, a, b, fa, fb, tol_f=tol, max_iter=max_iter, msign_steps=msign_steps)
    elif method == "secant":
        # Use bracket seeds if possible
        a, b, fa, fb, _ = bracket_find(G, Theta, 0.0, init_step=1.0, max_expand=60, msign_steps=msign_steps)
        lam0, lam1 = (a, b) if a != b else (0.0, 1.0)
        stats = secant_solve(G, Theta, lam0, lam1, tol_f=tol, max_iter=max_iter, msign_steps=msign_steps)
    elif method in ("fixed", "eq10"):
        stats = fixed_point_solve(G, Theta, lam0=0.0, tol_f=tol, max_iter=max_iter, msign_steps=msign_steps)
    elif method == "newton":
        # Prefer analytic derivative if dmsign is available
        use_numeric = dmsign_op is None
        stats = newton_solve(G, Theta, lam0=0.0, tol_f=tol, max_iter=max_iter, msign_steps=msign_steps,
                             numeric_derivative=use_numeric)
    else:
        raise ValueError(f"Unknown method: {method}")

    if verbose:
        print(
            f"[{stats.method}] lam={stats.lam:.10f} | |f|={stats.f_abs:.3e} "
            f"| iters={stats.iterations} | f_evals={stats.f_evals} | time={stats.time_sec*1e3:.1f} ms | converged={stats.converged}"
        )

    # Sanity: check f at solution and report phi-accuracy at solution vs reference
    f_sol, Phi_sol = f_value(G, Theta, stats.lam, msign_steps)
    Phi_ref_sol = phi_ref_polar(G + stats.lam * Theta)
    rel_phi_err_sol = float((Phi_sol - Phi_ref_sol).norm().item() / (Phi_ref_sol.norm().item() + 1e-12))
    if verbose:
        print(f"Check: f(lam)={f_sol:.6e}, rel_phi_err@lam={rel_phi_err_sol:.3e}")


def main():
    parser = argparse.ArgumentParser(description="Solve f(lam)=<Theta, msign(G+lam Theta)>=0 using multiple algorithms.")
    parser.add_argument("--method", type=str, default="brent", choices=["brent", "secant", "fixed", "eq10", "newton", "all"],
                        help="Root-finding method")
    parser.add_argument("--m", type=int, default=64, help="Row dimension of matrices")
    parser.add_argument("--n", type=int, default=32, help="Column dimension of matrices")
    parser.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--msign-steps", type=int, default=10, help="Iteration steps for msign operator")
    parser.add_argument("--tol", type=float, default=1e-8, help="Tolerance for |f|")
    parser.add_argument("--max-iters", type=int, default=100, help="Max iterations for solvers")
    parser.add_argument("--rank1-theta", action="store_true", help="Use rank-1 Theta (recommended)")
    parser.add_argument("--no-rank1-theta", dest="rank1_theta", action="store_false")
    parser.set_defaults(rank1_theta=True)
    parser.add_argument("--cases", type=int, default=1, help="Number of random cases to run")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    device = set_device(args.device)
    methods = [args.method]
    if args.method == "all":
        methods = ["brent", "secant", "fixed", "newton"]

    for k in range(args.cases):
        for mth in methods:
            run_one(
                method=mth,
                m=args.m,
                n=args.n,
                device=device,
                seed=args.seed + k,
                msign_steps=args.msign_steps,
                tol=args.tol,
                max_iter=args.max_iters,
                rank1_theta=args.rank1_theta,
                verbose=not args.quiet,
                compare_ref_phi=True,
            )


if __name__ == "__main__":
    main()
