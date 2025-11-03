# bench_msign.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import torch
from kernels.msign import msign  

# -----------------------------------------------------------------------------
# 环境与辅助
# -----------------------------------------------------------------------------
def torch_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()

def pick_device(cuda: bool) -> torch.device:
    if cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def set_determinism():
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.set_float32_matmul_precision("high")


# 计算一次 msign 迭代的大致 FLOPs（仅用于吞吐估算）
# 你的算法每步主要有：S = X @ X^T (r x c · c x r) -> ~2 r^2 c
#                         Y = (bI + cS) @ S          -> r x r · r x r -> ~2 r^3
#                         X_new = Y @ X              -> r x r · r x c -> ~2 r^2 c
# 总计每步 ~ (4 r^2 c + 2 r^3) 浮点乘加
def estimate_flops_per_step(shape: Tuple[int, int]) -> float:
    m, n = shape
    # msign 内部：若 rows > cols 会转置为 (n, m)，于是 r = min(m, n), c = max(m, n)
    r, c = (m, n) if m <= n else (n, m)
    return 4.0 * (r ** 2) * c + 2.0 * (r ** 3)

# -----------------------------------------------------------------------------
# 质量指标
# -----------------------------------------------------------------------------
@torch.no_grad()
def orthogonality_error(U: torch.Tensor) -> float:
    m, n = U.shape[-2], U.shape[-1]
    if m <= n:
        I = torch.eye(m, device=U.device, dtype=torch.float32)
        err = torch.linalg.norm(U @ U.mT - I, ord="fro") / math.sqrt(m)
    else:
        I = torch.eye(n, device=U.device, dtype=torch.float32)
        err = torch.linalg.norm(U.mT @ U - I, ord="fro") / math.sqrt(n)
    return float(err.item())

@torch.no_grad()
def polar_reconstruction_residual(G: torch.Tensor, U: torch.Tensor) -> float:
    # 以 U (U^T G) 作为极分解的“正交部分”投影重建，计算相对残差
    recon = U @ (U.mT @ G)
    num = torch.linalg.norm(G - recon, ord="fro")
    den = torch.linalg.norm(G, ord="fro").clamp_min(1e-20)
    return float((num / den).item())

@torch.no_grad()
def svd_polar_error(G: torch.Tensor, U_est: torch.Tensor) -> Optional[float]:
    # 仅小尺寸上做 SVD 基准
    m, n = G.shape[-2], G.shape[-1]
    try:
        U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
        U_svd = U @ Vh
        num = torch.linalg.norm(U_est.float() - U_svd, ord="fro")
        den = torch.linalg.norm(U_svd, ord="fro").clamp_min(1e-20)
        return float((num / den).item())
    except RuntimeError:
        return None

# -----------------------------------------------------------------------------
# 计时
# -----------------------------------------------------------------------------
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
        return start.elapsed_time(end) / 1000.0  # 秒
    # CPU
    t0 = time.perf_counter()
    func()
    t1 = time.perf_counter()
    return t1 - t0

@dataclass
class BenchConfig:
    shapes: List[Tuple[int, int]]
    steps_list: List[int]
    trials: int
    warmup: int
    cuda: bool
    seed: int
    svd_max_dim: int  # 仅当 max(m, n) <= 该阈值时做 SVD 基准

@dataclass
class BenchResult:
    shape: Tuple[int, int]
    steps: int
    mean_time_s: float
    std_time_s: float
    gflops: float
    ortho_err: float
    polar_resid: float
    svd_rel_err: Optional[float]

# -----------------------------------------------------------------------------
# 基准核心
# -----------------------------------------------------------------------------
def run_bench(cfg: BenchConfig) -> List[BenchResult]:
    set_determinism()
    device = pick_device(cfg.cuda)

    # 原来是 'cpu'，这里改成跟随实际 device
    gen = torch.Generator(device=device.type).manual_seed(cfg.seed)


    results: List[BenchResult] = []

    for (m, n) in cfg.shapes:
        # 生成高斯随机矩阵并略微“抬高”谱，减少病态
        G = torch.randn((m, n), generator=gen, device=device, dtype=torch.float32)
        # 可选：加一点对角偏置，缓解奇异情况（方阵时更有效）
        if m == n:
            G = G + 0.01 * torch.eye(m, device=device, dtype=torch.float32)

        for steps in cfg.steps_list:
            # 预热
            for _ in range(cfg.warmup):
                _ = msign(G, steps=steps)

            times: List[float] = []
            ortho_err: Optional[float] = None
            polar_resid: Optional[float] = None
            svd_err: Optional[float] = None

            for _ in range(cfg.trials):
                t = time_once(lambda: msign(G, steps=steps), device)
                times.append(t)
            # 额外再跑一次拿 U 用于指标
            U = msign(G, steps=steps)

            ortho_err = orthogonality_error(U)
            polar_resid = polar_reconstruction_residual(G, U)
            if max(m, n) <= cfg.svd_max_dim:
                svd_err = svd_polar_error(G, U)

            mean_t = float(torch.tensor(times).mean().item())
            std_t = float(torch.tensor(times).std(unbiased=False).item())

            # 估算吞吐
            flops = estimate_flops_per_step((m, n)) * steps  # 每步 FLOPs * steps
            gflops = flops / mean_t / 1e9

            results.append(
                BenchResult(
                    shape=(m, n),
                    steps=steps,
                    mean_time_s=mean_t,
                    std_time_s=std_t,
                    gflops=gflops,
                    ortho_err=ortho_err,
                    polar_resid=polar_resid,
                    svd_rel_err=svd_err,
                )
            )
    return results

def print_table(results: Sequence[BenchResult]) -> None:
    header = (
        f"{'shape':>12} | {'steps':>5} | {'time(ms)':>9} | {'std':>8} | "
        f"{'GFLOP/s':>9} | {'ortho_err':>10} | {'polar_resid':>11} | {'svd_rel_err':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        se = f"{r.svd_rel_err:.3e}" if r.svd_rel_err is not None else "   -   "
        print(
            f"{str(r.shape):>12} | {r.steps:>5d} | {r.mean_time_s:>9.4f} | {r.std_time_s:>8.4f} | "
            f"{r.gflops:>9.1f} | {r.ortho_err:>10.3e} | {r.polar_resid:>11.3e} | {se:>11}"
        )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_shape_list(s: str) -> List[Tuple[int, int]]:
    # 形如 "128x128,256x256,1024x1024,4096x4096" 或 "1024x2048"
    out = []
    for part in s.split(","):
        part = part.strip().lower().replace("*", "x")
        a, b = part.split("x")
        out.append((int(a), int(b)))
    return out

def main():
    parser = argparse.ArgumentParser(description="Benchmark for msign()")
    parser.add_argument(
        "--shapes",
        type=str,
        default="128x128,256x256,1024x1024,4096x4096",
        help="Comma-separated list like '128x128,256x256,1024x1024'",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="3,5,8",
        help="Comma-separated steps list, e.g. '3,5,8'",
    )
    parser.add_argument("--trials", type=int, default=10, help="Repeat times for timing")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (default: use CUDA if available)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--svd-max-dim", type=int, default=2048, help="Do SVD baseline only when max(m,n)<=this")
    args = parser.parse_args()

    cfg = BenchConfig(
        shapes=parse_shape_list(args.shapes),
        steps_list=[int(s) for s in args.steps.split(",") if s.strip()],
        trials=args.trials,
        warmup=args.warmup,
        cuda=not args.cpu,
        seed=args.seed,
        svd_max_dim=args.svd_max_dim,
    )

    results = run_bench(cfg)
    print_table(results)

if __name__ == "__main__":
    main()
