# Spectral-Ball λ Solver (GPU)

> **目标**：在谱球面约束（spectral-ball）场景下，快速、稳定地数值求解 `λ`，使  
> \[
> f(\lambda) \;=\; \langle \Theta, \Phi(\lambda) \rangle \;=\; 0,
> \qquad \Phi(\lambda) \;=\; \operatorname{msign}(G + \lambda \Theta).
> \]  
> 其中 \( \operatorname{msign}(M) = M (M^\top M)^{-1/2} \) 为**极分解的矩阵符号函数**。

本仓库实现了多种**根求解 / 固定点**算法，且**彻底 GPU 化**（无 SVD 主路径），用于验证 `λ` 的收敛速度、精度与实现效率。

---

## 目录结构

```
.
├── kernels/
│   ├── msign.py                 # 高精度/高效 Polar-Express msign 实现（BF16 内核 + 多项式迭代）
│   └── dmsign.py                # （可选）msign 的导数/VJP 实现（若提供，则 Newton 可用解析梯度）
│
├── solver/
│   ├── base.py                # 通用工具：FP32 累加内积/trace、双边幂迭代、统一目标评估等
│   ├── fix_point.py             # 固定点法 λ 求解（你提供的“完美风格”版本）
│   ├── brent.py                 # Brent 法（需 bracket）
│   ├── secant.py                # 割线法（纯函数值）
│   └── newton.py                # Newton 法（优先解析 VJP，无则用有限差分）
│
├── root_solver.py               # 统一入口：选择方法，打印可解析日志
│
├── scripts/
│   ├── check_each_solver.sh     # 小规模一键 sanity check（全方法）
│   ├── benchmark_methods.sh     # 多尺寸/多方法评测，汇总 CSV
│   ├── sweep_msign_steps.sh     # msign 迭代步数灵敏度评估
│   └── parse_logs_to_csv.sh     # 解析日志 → CSV
│
└── README.md
```

---

## 算法背景：为什么需要迭代法求解 λ？

我们考虑谱球面上的 Muon 型优化问题（几何直觉与一阶“切空间”近似来自相关系列文章）：

\[
\max_{\Phi} \;\langle G, \Phi \rangle
\quad \text{s.t.}\quad
\|\Phi\|_2 = 1, \;
\langle \Theta , \Phi \rangle = 0,\qquad
\Theta = u_1 v_1^\top,
\]

其中 \( \Theta \) 是当前参数 \( W \) 的谱范数梯度（即最大奇异值对应的左右奇异向量外积），可通过**双边幂迭代**高效获得（避免 SVD）。

通过拉格朗日乘子法，将约束嵌入目标函数，可得最优解形式为：

\[
\Phi(\lambda) \;=\; \operatorname{msign}(G + \lambda \Theta).
\]

于是原约束 \( \langle \Theta, \Phi(\lambda) \rangle = 0 \) 转化为非线性方程：

\[
f(\lambda) \;:=\; \langle \Theta, \operatorname{msign}(G + \lambda \Theta) \rangle \;=\; 0.
\]

由于 `msign(·)` 隐含矩阵平方根与逆运算，**该方程通常无解析解**，必须依赖**数值迭代法**求解 λ。本仓库提供了 4 种常用策略，覆盖无导数与有导数两类情形。

---

## 数学推导与各算法要点

### 统一记号

- \( Z = G + \lambda \Theta \)
- \( \Phi = \operatorname{msign}(Z) = Z (Z^\top Z)^{-1/2} \)
- \( X = \Theta^\top \Phi \)
- **关键恒等式**：  
  \[
  q \;:=\; Z^\top \Phi \;=\; (Z^\top Z)^{1/2}
  \]
  此恒等式使我们**无需显式计算矩阵平方根或逆**，只需 `q = Z^T Φ` 即可。
- 目标函数：\( f(\lambda) = \langle \Theta, \Phi(\lambda) \rangle = \mathrm{tr}(X) \)

#### 初始值（通用）
所有方法推荐使用同一初始值：
\[
\lambda_0 \;=\; - \frac{\mathrm{tr}(\Theta^\top G)}{\mathrm{tr}(\Theta^\top \Theta)}.
\]

---

### 1) 固定点法（`solver/fix_point.py`）

基于恒等式推导出的显式更新公式：
\[
\lambda \;=\; \frac{
  \mathrm{tr}(\Theta^\top \Phi \, q)
  - \frac{\mathrm{tr}(\Theta^\top \Phi)\,\mathrm{tr}(q)}{m}
  - \mathrm{tr}(\Theta^\top G)
}{
  \mathrm{tr}(\Theta^\top \Theta)
}.
\]

**实现细节**：
- 构造固定点迭代：\( \lambda_{k+1} \leftarrow \text{RHS}(\lambda_k) \)
- \( q = Z^\top \Phi \) 直接由 `msign` 输出计算，避免显式 `(Z^T Z)^{1/2}`
- **双重收敛准则**：  
  \( |f(\lambda)| = |\mathrm{tr}(X)| \) 与相对变化 \( |\Delta \lambda| / |\lambda| \) 同时小于阈值
- 使用 CUDA events 精确计时，日志包含 `ms_per_step`

---

### 2) 割线法（`solver/secant.py`）

经典无导数一元方程求根方法，仅需函数值。

**更新公式**：
\[
\lambda_{k+1} = \lambda_k - f(\lambda_k) \cdot \frac{\lambda_k - \lambda_{k-1}}{f(\lambda_k) - f(\lambda_{k-1})}.
\]

- 建议先通过 `find_bracket` 获取初始区间以提高稳定性
- 不保证全局收敛，但局部收敛较快

---

### 3) Brent 法（`solver/brent.py`）

融合二分法、割线法与逆二次插值的**稳健求根算法**。

- **要求**：存在区间 \([a, b]\) 使得 \( f(a) f(b) \leq 0 \)
- 在本实现中，函数评估在 GPU 上完成，标量逻辑在 CPU 侧执行
- 收敛性有理论保证，适合对鲁棒性要求高的场景

---

### 4) Newton 法（`solver/newton.py`）

若可获得导数 \( f'(\lambda) \)，则具有**二次收敛速度**。

**更新公式**：
\[
\lambda_{k+1} = \lambda_k - \frac{f(\lambda_k)}{f'(\lambda_k)}.
\]

- **优先使用解析梯度**：通过 `dmsign.py` 提供的 VJP 计算  
  \[
  f'(\lambda) = \langle \Theta, \; \mathrm{d}\Phi/\mathrm{d}A \,[\Theta] \rangle
  \]
- **回退方案**：若无解析导数，则使用中心差分近似 \( f'(\lambda) \)

---

## GPU 友好实现要点

- ✅ **`msign`**：采用 Polar-Express 多项式迭代，支持 BF16/FP32 混合精度，避免 SVD
- ✅ **`Θ` 构造**：通过双边幂迭代（仅 GEMM/GEMV）获取最大奇异向量
- ✅ **关键技巧**：利用 \( q = Z^\top \Phi \) 替代显式矩阵平方根
- ✅ **数值稳定**：标量运算尽量保留在 GPU 上，仅日志输出时转为 Python float
- ✅ **精确计时**：使用 CUDA events 统计每步耗时

---

## 评估指标与日志字段

所有方法通过 `root_solver.py` 输出统一格式日志，`scripts/parse_logs_to_csv.sh` 可将其解析为 CSV。关键字段如下：

| 字段 | 含义 |
|------|------|
| `lambda` | 最终解 \( \lambda^* \) |
| `abs_f` | \( |f(\lambda^*)| = |\mathrm{tr}(\Theta^\top \Phi)| \)，即**约束残差** |
| `abs_constraint` | 固定点法内部即时约束残差 |
| `iters` | 迭代次数 |
| `fevals` | 函数评估次数（Brent/Secant） |
| `time_ms` | 总耗时（毫秒） |
| `ortho_err` | \( \|\Phi^\top \Phi - I\|_F / m \)，用于诊断**半正交一致性** |
| `bracket_lo/hi` | Brent/Secant 的初始括号区间（若存在） |

这些指标帮助你在**收敛性**、**精度**与**效率**之间进行权衡：
- **固定点法**：步数少、单步贵（依赖 `msign`），但完全 GPU 化
- **Brent 法**：稳健可靠，适合黑盒场景
- **Newton 法**：若有解析梯度，通常收敛最快

---

## 安装与运行

### 前置条件
- Python ≥ 3.8
- PyTorch + CUDA（确保 `torch.cuda.is_available()` 为 `True`）

### 快速开始
```bash
# 赋予脚本执行权限
chmod +x scripts/*.sh

# 1. 快速自测（小规模，全方法）
bash scripts/check_each_solver.sh

# 2. 批量评测（多尺寸/多方法，生成 CSV）
bash scripts/benchmark_methods.sh
# 结果保存至: results/benchmarks.csv

# 3. msign 迭代步数灵敏度分析
bash scripts/sweep_msign_steps.sh
# 结果保存至: results/sweep_msign_steps.csv
```

### 自定义参数
所有脚本支持通过环境变量覆盖默认值，例如：
```bash
SIZES="128x64 256x128" \
METHODS="fixed_point newton" \
SEED=7 \
TOL=1e-6 \
MAX_ITER=50 \
MSIGN_STEPS=7 \
bash scripts/benchmark_methods.sh
```

---

## 复现注意事项

- **随机性控制**：`root_solver.py` 通过 `--seed` 控制 `G` 和 `W` 的生成；`Θ` 由 `W` 的最大奇异向量决定
- **dmsign 可选**：若提供 `kernels/dmsign.py`，Newton 法将使用解析梯度；否则回退至有限差分
- **数值稳定性**：全程使用 FP32 累加；`msign` 内部包含安全缩放与 NaN/Inf 兜底处理

---

## 引用与参考

- “流形上的最速下降”系列文章（谱球面 / 正交 / Stiefel 约束优化）
- **Polar Express**: 高效矩阵函数的多项式迭代方法（[arXiv:2505.16932](https://arxiv.org/abs/2505.16932)）
- **数值分析标准方法**: Brent, Secant, Newton 求根算法
