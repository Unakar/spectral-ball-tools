from __future__ import annotations
from typing import Tuple, Dict, List
import time
import torch

# -----------------------------
# Power Iteration Implementations
# -----------------------------
@torch.no_grad()
def baseline_pi(w: torch.Tensor, steps: int = 10):
    w = w.to(torch.float32)
    v = torch.ones_like(w[..., :1, :].transpose(-2, -1))
    for _ in range(steps):
        v = torch.nn.functional.normalize(w.transpose(-2, -1) @ (w @ v), dim=-2)
    u = torch.nn.functional.normalize(w @ v, dim=-2)
    s = (u.transpose(-2, -1) @ w @ v).squeeze(-1).squeeze(-1)
    return s, u, v

#optimized_pi_v1
# == Size: 1024 x 1024 (avg over 20 trials) ==
# Method          Time (ms)    σ_rel_err     uv_F_err
# Baseline PI         4.643    4.883e-03    9.689e-01
# Optimized PI        3.225    4.883e-03    9.689e-01
# SVD                80.186          0.0          0.0
# @torch.no_grad()
# def optimized_pi_v1(w: torch.Tensor, steps: int = 10):
#     w = w.to(torch.float32)
#     # 预计算 W^T W
#     w_t_w = w.transpose(-2, -1) @ w
    
#     v = torch.ones_like(w[..., :1, :].transpose(-2, -1))
#     for _ in range(steps):
#         # 使用预计算的 W^T W 和 F.normalize 融合
#         v = torch.nn.functional.normalize(w_t_w @ v, dim=-2)
    
#     u = torch.nn.functional.normalize(w @ v, dim=-2)
#     s = (u.transpose(-2, -1) @ w @ v).squeeze(-1).squeeze(-1)
#     return s, u, v


# optimized_pi_v2
# == Size: 1024 x 1024 (avg over 20 trials) ==
# Method          Time (ms)    σ_rel_err     uv_F_err
# Baseline PI         4.786    4.883e-03    9.689e-01
# Optimized PI        2.456    4.883e-03    9.689e-01
# SVD                80.034          0.0          0.0
@torch.no_grad()
def optimized_pi(w: torch.Tensor, steps: int = 10, eps: float = 1e-20):
    w = w.to(torch.float32)
    gram = w.transpose(-2, -1).matmul(w) #预计算W^Tw
    v = torch.ones(*w.shape[:-2], w.shape[-1], 1, device=w.device, dtype=w.dtype)
    for _ in range(steps):
        v = gram.matmul(v)
        v = v / torch.clamp(torch.linalg.vector_norm(v, ord=2, dim=-2, keepdim=True), min=eps)
    u = w.matmul(v)
    u = u / torch.clamp(torch.linalg.vector_norm(u, ord=2, dim=-2, keepdim=True), min=eps)
    s = (u.transpose(-2, -1).matmul(w).matmul(v)).squeeze(-1).squeeze(-1)
    return s, u, v


@torch.no_grad()
def top_svd(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    W = W.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    u1 = U[:, :1]
    s1 = S[0]
    v1 = Vh[:1, :].t()
    return s1, u1, v1


@torch.no_grad()
def triple_metrics(
    s: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    s_true: torch.Tensor,
    u_true: torch.Tensor,
    v_true: torch.Tensor
) -> Dict[str, float]:
    eps = 1e-20
    sigma_rel_error = float(torch.abs(s - s_true) / (torch.abs(s_true) + eps))
    M_est = u @ v.t()
    M_true = u_true @ v_true.t()
    frob_diff = torch.linalg.norm(M_est - M_true, ord='fro')
    frob_true = torch.linalg.norm(M_true, ord='fro')
    uv_frob_rel_error = float(frob_diff / (frob_true + eps))
    return dict(sigma_rel_error=sigma_rel_error, uv_frob_rel_error=uv_frob_rel_error)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, *args, **kwargs):
    _sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _sync()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return dt_ms, out


# -----------------------------
# Unified Benchmark
# -----------------------------
@torch.no_grad()
def benchmark(
    sizes: List[int],
    trials: int = 5,
    pi_steps: int = 20,
    add_shift: float = 0.1,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}, trials={trials}, power-steps={pi_steps}")

    # Warmup
    dummy = torch.randn(64, 64, device=device)
    _timed(baseline_pi, dummy, 5)
    _timed(optimized_pi, dummy, 5)
    _timed(top_svd, dummy)

    for n in sizes:
        # Accumulators
        times_baseline, times_optimized, times_svd = [], [], []
        sigma_err_bl, sigma_err_opt = [], []
        uv_err_bl, uv_err_opt = [], []

        for _ in range(trials):
            W = torch.randn(n, n, device=device, dtype=torch.float32)
            if add_shift:
                W = W + add_shift * torch.eye(n, device=device, dtype=torch.float32)

            # SVD (ground truth)
            t_svd, (s_true, u_true, v_true) = _timed(top_svd, W)
            times_svd.append(t_svd)

            # Baseline PI
            t_bl, (s_bl, u_bl, v_bl) = _timed(baseline_pi, W, pi_steps)
            times_baseline.append(t_bl)
            m_bl = triple_metrics(s_bl, u_bl, v_bl, s_true, u_true, v_true)
            sigma_err_bl.append(m_bl["sigma_rel_error"])
            uv_err_bl.append(m_bl["uv_frob_rel_error"])

            # Optimized PI
            t_opt, (s_opt, u_opt, v_opt) = _timed(optimized_pi, W, pi_steps)
            times_optimized.append(t_opt)
            m_opt = triple_metrics(s_opt, u_opt, v_opt, s_true, u_true, v_true)
            sigma_err_opt.append(m_opt["sigma_rel_error"])
            uv_err_opt.append(m_opt["uv_frob_rel_error"])

        # Averages
        avg = lambda x: sum(x) / len(x)
        t_bl_avg = avg(times_baseline)
        t_opt_avg = avg(times_optimized)
        t_svd_avg = avg(times_svd)
        sigma_bl = avg(sigma_err_bl)
        sigma_opt = avg(sigma_err_opt)
        uv_bl = avg(uv_err_bl)
        uv_opt = avg(uv_err_opt)

        # Print unified table
        print(f"\n== Size: {n} x {n} (avg over {trials} trials) ==")
        print(f"{'Method':<14} {'Time (ms)':>10}  {'σ_rel_err':>11}  {'uv_F_err':>11}")
        print(f"{'Baseline PI':<14} {t_bl_avg:10.3f}  {sigma_bl:11.3e}  {uv_bl:11.3e}")
        print(f"{'Optimized PI':<14} {t_opt_avg:10.3f}  {sigma_opt:11.3e}  {uv_opt:11.3e}")
        print(f"{'SVD':<14} {t_svd_avg:10.3f}  {0.0:11.1f}  {0.0:11.1f}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    sizes = [128, 256, 1024]
    benchmark(sizes=sizes, trials=20, pi_steps=100, add_shift=0.1, seed=0)