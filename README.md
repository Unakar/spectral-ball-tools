# Spectral-Ball λ Solver (GPU)

目标：在谱球面约束（spectral-ball）下求解标量 λ，使

f(λ) = ⟨Θ, Φ(λ)⟩ = 0，  Φ(λ) = msign(G + λΘ)

其中 msign(M) = M (MᵀM)^{-1/2} 是极分解的矩阵符号函数。本仓库使用 Polar‑Express 的多项式迭代在 GPU 上高效近似该运算（SVD‑free，BF16/FP32 混合精度）。

提供多种一元求根/固定点算法，统一只以 |f(λ)| ≤ tol 作为收敛判据，便于比较不同方法的收敛速度、精度与效率。

---

## 目录

```
.
├── kernels/
│   ├── msign.py               # 高精度/高效 msign（Polar-Express, BF16/FP32）
│   ├── dmsign.py              # （可选）msign 的 VJP/导数实现（供 Newton 使用）
│   └── power_iteration.py     # 双边幂迭代 power_iteration；top_svd 基准
│
├── solver/
│   ├── base.py                # compute_f/compute_phi 等通用工具（与方法解耦）
│   ├── bisection.py           # 纯二分法（需 monotone + bracket）
│   ├── brent.py               # Brent 法（需 bracket）
│   ├── secant.py              # 割线法（无导数）
│   ├── newton.py              # Newton 法（优先解析导数，回退差分）
│   └── fix_point.py           # 固定点法（显式更新，SVD-free）
│
├── root_solver.py             # 统一入口：选择方法/Θ构造，输出日志
└── scripts/
    ├── check_each_solver.sh   # 小规模一键 sanity check（全方法）
    ├── becnmark_methods.sh    # 多尺寸/多方法评测，汇总 CSV（文件名有拼写）
    ├── sweep_msign_steps.sh   # msign 迭代步数灵敏度评估
    └── parse_logs_to_csv.sh   # 解析日志 → CSV
```

---

## 快速开始

单次运行（选择方法 + Θ 构造）：
- 关键参数：
  - `--method {brent,bisection,secant,fixed_point,newton}`
  - `--n N --m M`：矩阵尺寸
  - `--seed S`：随机种子
  - `--tol T`：收敛阈值（仅以 |f(λ)| 判断）
  - `--max_iter K`：最大迭代步数
  - `--msign_steps S`：msign 迭代步数（越大越准、越慢）
  - `--theta_source {power,svd}`：Θ 的构造方式（默认 power）
  - `--power_iters P`：幂迭代步数（用于 `theta_source=power`）

示例：
```
python -u root_solver.py \
  --method brent \
  --n 128 --m 256 \
  --seed 42 \
  --tol 1e-8 \
  --max_iter 100 \
  --msign_steps 5 \
  --theta_source power \
  --power_iters 3
```

脚本：
```

# 1) 小规模全方法自测
bash scripts/check_each_solver.sh

# 2) 多尺寸/多方法评测并汇总 CSV（注意文件名是 becnmark_methods.sh）
SIZES="128x64 256x128" METHODS="bisection brent newton" \
THETA_SOURCE=power POWER_ITERS=3 TOL=1e-8 MAX_ITER=100 MSIGN_STEPS=5 \
bash scripts/becnmark_methods.sh

# 3) msign 步数灵敏度
METHOD=fixed_point N=128 M=256 THETA_SOURCE=power POWER_ITERS=3 \
bash scripts/sweep_msign_steps.sh
```

---

## 日志字段与判据

统一由 `root_solver.py` 打印：
- `f(0)`：λ=0 时的函数值（便于了解初始偏差）。
- `orthogonality error @λ`：半正交一致性误差 `‖ΦᵀΦ − I‖_F / m`（诊断 msign 质量）。
- 结果：
  - `λ*`：最终解
  - `|f(λ*)|`：残差（唯一收敛判据，≤ `--tol` 即视为收敛）
  - `iters`：迭代步数（唯一统计的步数指标）
  - `converged`：是否满足收敛
  - `time`：求解耗时（毫秒）
  - `bracket`：如适用，显示括号区间

CSV（`scripts/parse_logs_to_csv.sh`）会将日志解析为表格，字段包含：`lambda, abs_f, iters, time_ms, ortho_err, bracket_lo/hi` 等。

---

## 求解器简介

- 固定点法 `solver/fix_point.py`
  - 显式更新：λ ← [ tr(Xq) − tr(X)·tr(q)/m − tr(ΘᵀG) ] / tr(ΘᵀΘ)，其中 X=ΘᵀΦ, q=ZᵀΦ；全流程 GPU。

- 二分法 `solver/bisection.py`
  - 单调性来自 ∥G+λΘ∥_* 的凸性；需括号区间；线性收敛，鲁棒。

- Brent 法 `solver/brent.py`
  - 需括号区间；融合二分/割线/逆二次插值，在保证性的同时更快。

- 割线法 `solver/secant.py`
  - 无导数；使用最近两步值做线性外推；常配合括号获取初始两点。

- Newton 法 `solver/newton.py`
  - 有导数时收敛快：优先用 `kernels/dmsign.py` 的 VJP，缺少时回退中心差分。

---

## 记号与实现要点（简）

- 记号：Z=G+λΘ，Φ=msign(Z)，X=ΘᵀΦ，q=ZᵀΦ=(ZᵀZ)^{1/2}
- 关键：q=ZᵀΦ 避免显式矩阵平方根/逆
- Θ 构造：`--theta_source power|svd`（power 仅 GEMM/GEMV；svd 供基准）
- msign：Polar‑Express 多项式迭代，BF16/FP32 混合精度；`--msign_steps` 控制精度/速度
- 数值：内积/trace 统一 FP32 累加；仅打印时转 Python float

---

## 复现与扩展

- 随机性：通过 `--seed` 控制；Θ 由 W 的最大奇异向量外积得到
- 解析梯度：提供 `kernels/dmsign.py` 则 Newton 使用解析 VJP；否则自动用差分
- 扩展新方法：实现 `solve_with_xxx(G, Θ, ..., msign_steps)` 并返回 `SolverResult`；用 `compute_f/compute_phi` 即可

---

## 背景简介

从拉格朗日乘子推导可得 Φ(λ)=msign(G+λΘ)，从而 f(λ)=⟨Θ,Φ(λ)⟩=0。由于 msign(·) 隐含矩阵平方根/逆，通常无解析解，需数值法求 λ。我们的实现彻底 GPU 化，避免 SVD 主路径，便于在实际深度学习/几何优化中使用与评测。

