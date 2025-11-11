# bench_msign.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import torch
from kernels.msign import msign  # 需保证可导入

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
    r, c = (m, n) if m <= n else (n, m)
    return 4.0 * (r ** 2) * c + 2.0 * (r ** 3)

# -----------------------------------------------------------------------------
# 质量指标
# -----------------------------------------------------------------------------
@torch.no_grad()
def orthogonality_error(U: torch.Tensor) -> float:
    U = U.float()
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
    U = U.float()
    G = G.float()
    recon = U @ (U.mT @ G)
    num = torch.linalg.norm(G - recon, ord="fro")
    den = torch.linalg.norm(G, ord="fro").clamp_min(1e-20)
    return float((num / den).item())

@torch.no_grad()
def svd_polar_error(G: torch.Tensor, U: torch.Tensor) -> Optional[float]:
    Gf = G.float()
    U_est = U.float()
    try:
        U_s, S, Vh = torch.linalg.svd(Gf, full_matrices=False)
        Q = U_s @ Vh
        num = torch.linalg.norm(U_est - Q, ord="fro")
        den = torch.linalg.norm(Q, ord="fro").clamp_min(1e-20)
        return float((num / den).item())
    except RuntimeError:
        return None

# 新增：对 U 做 SVD 的奇异值一致性指标
@torch.no_grad()
def singular_values_metrics(U: torch.Tensor) -> Tuple[float, float, float]:
    """
    返回 (sv_ones_dist, sv_max, sv_min)
    - sv_ones_dist: ||sigma - 1||_2 的向量范数（等价于 Frobenius on diag），
      其中 sigma 为 U 的非零奇异值向量（tight/compact SVD 的奇异值）。
    - sv_max: 最大奇异值
    - sv_min: 最小奇异值（当 U 非方阵时，是紧致SVD下的最小非零奇异值）
    """
    U = U.float()
    # 紧致 SVD 更符合“非零奇异值应为1”的检查
    try:
        _, S, _ = torch.linalg.svd(U, full_matrices=False)
    except RuntimeError:
        # 退化到 eig-based 或返回 NaN
        return float("nan"), float("nan"), float("nan")

    ones = torch.ones_like(S)
    sv_ones_dist = torch.linalg.norm(S - ones, ord=2).item()  # 向量2范数
    sv_max = S.max().item()
    sv_min = S.min().item()
    return float(sv_ones_dist), float(sv_max), float(sv_min)

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
    sv_ones_dist: float
    sv_max: float
    sv_min: float

# -----------------------------------------------------------------------------
# 基准核心
# -----------------------------------------------------------------------------
def run_bench(cfg: BenchConfig) -> List[BenchResult]:
    # set_determinism()
    device = pick_device(cfg.cuda)

    gen = torch.Generator(device=device.type).manual_seed(cfg.seed)

    results: List[BenchResult] = []

    for (m, n) in cfg.shapes:
        # 随机 G，方阵时稍抬谱
        G = torch.randn((m, n), generator=gen, device=device, dtype=torch.float32)
        if m == n:
            G = G + 0.01 * torch.eye(m, device=device, dtype=torch.float32)

        for steps in cfg.steps_list:
            # 预热
            for _ in range(cfg.warmup):
                _ = msign(G, steps=steps)

            times: List[float] = []
            for _ in range(cfg.trials):
                t = time_once(lambda: msign(G, steps=steps), device)
                times.append(t)

            # 指标
            U = msign(G, steps=steps)

            U_f32 = U.float()
            G_f32 = G.float()

            ortho_err = orthogonality_error(U_f32)
            polar_resid = polar_reconstruction_residual(G_f32, U_f32)
            svd_err: Optional[float] = None
            if max(m, n) <= cfg.svd_max_dim:
                svd_err = svd_polar_error(G_f32, U_f32)

            sv_ones_dist, sv_max, sv_min = singular_values_metrics(U_f32)

            mean_t = float(torch.tensor(times).mean().item())
            std_t = float(torch.tensor(times).std(unbiased=False).item())

            # 吞吐（估算）
            flops = estimate_flops_per_step((m, n)) * steps
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
                    sv_ones_dist=sv_ones_dist,
                    sv_max=sv_max,
                    sv_min=sv_min,
                )
            )
    return results

def print_table(results: Sequence[BenchResult]) -> None:
    header = (
        f"{'shape':>12} | {'steps':>5} | {'time(s)':>8} | {'std':>8} | "
        f"{'GFLOP/s':>9} | {'ortho_err':>10} | {'polar_resid':>11} | {'svd_rel_err':>11} | "
        f"{'sv||1-s||':>10} | {'sv_max':>8} | {'sv_min':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        se = f"{r.svd_rel_err:.3e}" if r.svd_rel_err is not None else "   -   "
        print(
            f"{str(r.shape):>12} | {r.steps:>5d} | {r.mean_time_s:>8.4f} | {r.std_time_s:>8.4f} | "
            f"{r.gflops:>9.1f} | {r.ortho_err:>10.3e} | {r.polar_resid:>11.3e} | {se:>11} | "
            f"{r.sv_ones_dist:>10.3e} | {r.sv_max:>8.4f} | {r.sv_min:>8.4f}"
        )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_shape_list(s: str) -> List[Tuple[int, int]]:
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