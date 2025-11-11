# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.msign import msign as msign_ours


def torch_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def pick_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_no_tf32() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"


@torch.no_grad()
def svd_polar_error(G: torch.Tensor, U_est: torch.Tensor) -> Optional[float]:
    m, n = G.shape[-2], G.shape[-1]
    try:
        U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
        U_svd = U @ Vh
        num = torch.linalg.norm(U_est.float() - U_svd, ord="fro")
        den = torch.linalg.norm(U_svd, ord="fro").clamp_min(1e-20)
        return float((num / den).item())
    except RuntimeError:
        return None


@torch.no_grad()
def time_once(func, device: torch.device) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch_sync(device)
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000.0
    t0 = time.perf_counter()
    func()
    t1 = time.perf_counter()
    return t1 - t0


MUON_COEFFICIENT_SETS = {
    "polar_express": [
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
    ],
}


def _muon_newton_schulz_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X


def muon_newton_schulz(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str = "polar_express",
    eps: float = 1e-7,
    transpose: Optional[bool] = None,
) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("x must have at least 2 dims")
    if x.dtype != torch.float32:
        x = x.float()

    if transpose is None:
        transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    X = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)

    if coefficient_type not in MUON_COEFFICIENT_SETS:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}")
    coeffs = MUON_COEFFICIENT_SETS[coefficient_type]
    if steps % len(coeffs) != 0:
        raise ValueError(
            f"steps ({steps}) must be a multiple of coefficient cycle length ({len(coeffs)}) for '{coefficient_type}'."
        )

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        X = _muon_newton_schulz_step(X, a, b, c)

    if transpose:
        X = X.mT
    return X


@dataclass
class BenchCfg:
    shapes: List[Tuple[int, int]]
    steps_list: List[int]
    trials: int
    warmup: int
    cuda: bool
    seed: int
    svd_max_dim: int
    compile_both: bool


@dataclass
class Row:
    method: str
    shape: Tuple[int, int]
    steps: int
    time_mean_s: float
    time_std_s: float
    svd_rel_err: Optional[float]


def run_compare(cfg: BenchCfg) -> List[Row]:
    set_no_tf32()
    device = pick_device(cfg.cuda)
    gen = torch.Generator(device=device.type).manual_seed(cfg.seed)

    cycle_len = len(MUON_COEFFICIENT_SETS["polar_express"])
    invalid = [s for s in cfg.steps_list if s % cycle_len != 0]
    if invalid:
        raise ValueError(
            f"steps {invalid} are not multiples of the coefficient cycle length {cycle_len} for 'polar_express'. "
            f"Use values like {cycle_len}, {2*cycle_len}, ..."
        )

    out: List[Row] = []
    polarexpress_fn = msign_ours
    muon_fn = muon_newton_schulz
    if cfg.compile_both:
        try:
            muon_fn = torch.compile(muon_newton_schulz)
        except Exception:
            muon_fn = muon_newton_schulz

    for (m, n) in cfg.shapes:
        G = torch.randn((m, n), generator=gen, device=device, dtype=torch.float32)
        if m == n:
            G = G + 0.01 * torch.eye(m, device=device, dtype=torch.float32)

        for steps_req in cfg.steps_list:
            # Ours
            for _ in range(cfg.warmup):
                _ = polarexpress_fn(G, steps=steps_req)

            times: List[float] = []
            for _ in range(cfg.trials):
                t = time_once(lambda: polarexpress_fn(G, steps=steps_req), device)
                times.append(t)
            U = polarexpress_fn(G, steps=steps_req)

            mean_t = float(torch.tensor(times).mean().item())
            std_t = float(torch.tensor(times).std(unbiased=False).item())

            out.append(
                Row(
                    method="polarexpress",
                    shape=(m, n),
                    steps=steps_req,
                    time_mean_s=mean_t,
                    time_std_s=std_t,
                    svd_rel_err=svd_polar_error(G, U) if max(m, n) <= cfg.svd_max_dim else None,
                )
            )

            # Muon
            steps_muon = steps_req
            for _ in range(cfg.warmup):
                _ = muon_fn(G, steps=steps_muon)

            times = []
            for _ in range(cfg.trials):
                t = time_once(lambda: muon_fn(G, steps=steps_muon), device)
                times.append(t)
            U = muon_fn(G, steps=steps_muon)

            mean_t = float(torch.tensor(times).mean().item())
            std_t = float(torch.tensor(times).std(unbiased=False).item())

            out.append(
                Row(
                    method="magatronmuon",
                    shape=(m, n),
                    steps=steps_muon,
                    time_mean_s=mean_t,
                    time_std_s=std_t,
                    svd_rel_err=svd_polar_error(G, U) if max(m, n) <= cfg.svd_max_dim else None,
                )
            )

    return out


def print_table(rows: Sequence[Row]) -> None:
    method_order = {"polarexpress": 0, "magatronmuon": 1}
    rows = sorted(rows, key=lambda r: (r.shape[0], r.shape[1], r.steps, method_order.get(r.method, 9)))
    header = (
        f"{'method':>12} | {'shape':>12} | {'steps':>5} | "
        f"{'time(ms)':>9} | {'std(ms)':>9} | {'svd_rel_err':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        se = f"{r.svd_rel_err:.3e}" if r.svd_rel_err is not None else "   -   "
        print(
            f"{r.method:>12} | {str(r.shape):>12} | {r.steps:>5d} | "
            f"{(r.time_mean_s*1e3):>9.2f} | {(r.time_std_s*1e3):>9.2f} | {se:>11}"
        )


def parse_shape_list(s: str) -> List[Tuple[int, int]]:
    out = []
    for part in s.split(","):
        part = part.strip().lower().replace("*", "x")
        a, b = part.split("x")
        out.append((int(a), int(b)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare polarexpress vs magatronmuon (NS) speed/accuracy")
    parser.add_argument(
        "--shapes",
        type=str,
        default="256x256,128*2048",
        help="Comma-separated list like '128x128,256x256,1024x1024'",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="8",
        help="Comma-separated steps list. Note: magatronmuon requires steps to be a multiple of its coefficient cycle length (e.g., 8 for polar_express).",
    )
    parser.add_argument("--trials", type=int, default=10, help="Repeat times for timing")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs (JIT, cache warmup)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (default: CUDA if available)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--svd-max-dim",
        type=int,
        default=2048,
        help="Compute SVD baseline only when max(m,n) <= this threshold",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile; by default both methods are compiled for fairness.",
    )

    args = parser.parse_args()

    cfg = BenchCfg(
        shapes=parse_shape_list(args.shapes),
        steps_list=[int(s) for s in args.steps.split(",") if s.strip()],
        trials=args.trials,
        warmup=args.warmup,
        cuda=not args.cpu,
        seed=args.seed,
        svd_max_dim=args.svd_max_dim,
        compile_both=not args.no_compile,
    )

    rows = run_compare(cfg)
    print_table(rows)


if __name__ == "__main__":
    main()