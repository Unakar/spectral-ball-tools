# solve_lambda.py 使用说明与算法说明

本脚本用于求解标量方程

  f(λ) = <Θ, msign(G + λ Θ)> = 0

其中 msign(·) 为极分解的单位因子（polar unitary factor），即 A = U P，U = msign(A)，P = (A^T A)^{1/2}。脚本集成了多种一维根求解算法，便于比较收敛性与数值精度。

本文档介绍：
- 算法原理与来龙去脉
- 代码结构与实现要点
- 命令行使用方法与实验建议

----------------------------------------

## 问题形式与记号

- 矩阵 G ∈ R^{m×n}，方向矩阵 Θ ∈ R^{m×n}（默认生成 rank-1）。
- A(λ) = G + λ Θ，Φ(λ) = msign(A(λ))。
- 目标方程：f(λ) = <Θ, Φ(λ)> = 0（Frobenius 内积）。

msign 的高效实现位于 `dev/zhihu/msign.py`，采用多项式迭代（Polar Express 风格）。

----------------------------------------

## 算法概览

脚本中提供四种互补的求解策略：

- Brent 法（brent）：全局稳健，基于括号区间融合二分、割线、逆二次插值。
- 割线法（secant）：无导数，局部收敛，作为备选或回退方案。
- 固定点法（fixed，兼容别名 eq10）：将“公式(10)”解释为预条件残差下降 λ <- λ − c(λ) f(λ)，线性收敛；实现中带阻尼与回溯以抑制震荡。
- 牛顿法（newton）：使用 dmsign 的 VJP（伴随）计算 f'(λ)，通常迭代步数最少；当 dmsign 不可用时退回有限差分近似导数。

后文逐一说明。

----------------------------------------

## 固定点法（fixed，原 eq10）

思路：将“公式(10)”重写为预条件的残差迭代

  λ_{k+1} = λ_k − c(λ_k) · f(λ_k)

其中 c(λ) ≈ tr(P)/(m · tr(Θ^T Θ))，P = (A^T A)^{1/2}，tr(P) 即 A 的核范数（奇异值之和），m 为行数。该方法本质是对标量残差 f(λ) 的一维梯度下降（柯西步），局部线性收敛；若谱病态或步长失配可能震荡或变慢。

实现要点：
- 函数：`fixed_point_solve`（`solve_lambda.py`）。
- 步长系数：`eq10_step_c(A, Θ)`，内部用 `nuclear_norm(A)` 计算 tr(P)。
- 数值稳健：带 backtracking（减半步长）以避免残差上升；支持 `--msign-steps` 调整 msign 精度。

适用：可作 warm‑start 或轻量修正器；主力建议搭配 Brent 或 Newton。

----------------------------------------

## Brent 法（brent）

思路：在括号 [a,b] 上融合二分、割线和逆二次插值，具有全局稳健性和较快的收敛速度。只需能评估 f(λ)。

实现要点：
- `bracket_find`：对称扩张搜索找 [a,b] 使 f(a)·f(b)≤0。
- `brent_solve`：标准 Brent-Dekker 实现，自动在插值与二分之间切换。
- 统计：记录迭代数、函数评估次数、耗时毫秒，便于横向比较。

适用：可靠稳健的默认选择。

----------------------------------------

## 割线法（secant）

思路：用两点近似导数，更新 λ_{k+1} = λ_k − f(λ_k)·(λ_k−λ_{k−1})/(f(λ_k)−f(λ_{k−1}))，无需导数。

实现要点：
- `secant_solve`：若括号失败可用端点作为初值补救；没有全局保证，速度依赖初值。

适用：导数难得或仅需简易对比时。

----------------------------------------

## 牛顿法（newton）

思路：需要 f'(λ)。对 f(λ) = tr(Θ^T · msign(A(λ)))，

  f'(λ) = <Θ, D msign_{A(λ)}[Θ]>

我们采用 VJP（伴随）形式计算这个标量导数，避免显式构造 JVP：

- 设 O = msign(A)，A1 = A O^T（m×m），B1 = O^T A（n×n）。
- 构造块矩阵 M = [[A1 + εI, −C],[0, −(B1 + εI)]]，其中 C = Θ。
- 计算 S = mcsgn(M)（方阵上的矩阵符号，使用与 msign 相同风格的多项式迭代）。
- 读出 Sylvester 解 X = −1/2 · S[:m, m:]。
- VJP 给出：dA = X − O X^T O，则 f'(λ) = <dA, Θ>。

实现要点：
- `dev/zhihu/dmsign.py` 提供 `dmsign_vjp(A, C, msign_fn)`，完全基于已有的 `msign` 与 `mcsgn`，不需要 SVD/EVD。
- `solve_lambda.py` 中的 `newton_solve` 优先调用 `dmsign_vjp` 求导；不可用时退回中心差分近似。
- 目前未加线搜索/保括号的全局化包装，如需要可扩展。

适用：当 dmsign_vjp 可用时通常迭代步数最少。

----------------------------------------

## 代码结构（solve_lambda.py）

- 动态加载
  - 从同目录加载 `msign.py` 的 `msign`。
  - 从 `dmsign.py` 加载 `dmsign_vjp`（可选）。

- 工具与基元
  - `set_device`：选择 CPU/GPU。
  - `frob_inner`：Frobenius 内积。
  - `nuclear_norm`：核范数（奇异值和）。
  - `phi_ref_polar`：参考极分解单位因子（仅用于误差评估）。
  - `f_value(G, Θ, λ)`：计算 f(λ) 与 Φ(λ)。
  - `eq10_step_c(A, Θ)`：固定点步长系数 c(λ)。

- 求解器
  - `fixed_point_solve`（原 eq10）：固定点迭代，带回溯。
  - `bracket_find` + `brent_solve`：括号 + Brent。
  - `secant_solve`：割线法。
  - `newton_solve`：优先用 `dmsign_vjp` 求导，否则差分。
  - `SolveStats`：记录 method、λ、|f|、iters、f_evals、time(ms)、是否收敛。

- 数据与主流程
  - `generate_case`：随机生成 (G, Θ)，默认 Θ 为 rank‑1 并归一化。
  - `run_one`：跑单例，用参考极分解对比 `msign` 的相对误差；打印统计。
  - `main`：命令行参数解析与批量运行。

----------------------------------------

## 使用说明（CLI）

基本用法：

- Brent（默认）：
  - `python3 dev/zhihu/solve_lambda.py --method brent --m 64 --n 32 --seed 0 --device auto`
- 固定点法（原 eq10，现推荐名 fixed）：
  - `python3 dev/zhihu/solve_lambda.py --method fixed --m 64 --n 32`
- Newton（若 dmsign_vjp 可用会用解析导数，否则差分）：
  - `python3 dev/zhihu/solve_lambda.py --method newton --m 64 --n 32`
- 依次跑全套：
  - `python3 dev/zhihu/solve_lambda.py --method all --m 64 --n 32 --seed 0`

常用参数：

- `--m/--n`：矩阵尺寸。
- `--device`：`cpu|cuda|auto`。
- `--seed`：随机种子；`--cases` 可跑多个样本。
- `--msign-steps`：msign 迭代次数（默认 10）。
- `--tol`：|f| 收敛阈值（默认 1e-8）。
- `--max-iters`：最大迭代步数（各方法）。
- `--rank1-theta/--no-rank1-theta`：是否用 rank‑1 的 Θ（默认启用）。

输出说明：

- 首行：`f(0)` 与 `rel_phi_err@0`（msign 与参考极分解单位因子的相对误差）。
- 方法行：`[method] lam=... | |f|=... | iters=... | f_evals=... | time=... ms | converged=...`
- 末行：`f(lam)` 与 `rel_phi_err@lam` 的检查。

----------------------------------------

## 实验建议与调参

- 推荐默认：Brent（稳健）。
- Newton：当 `dmsign_vjp` 可用时很高效；病态例子可配线搜索/保括号（可扩展）。
- 固定点（fixed）：可作为 warm‑start；在谱病态时可能震荡，已带回溯减轻。
- `--msign-steps`：8–12 之间调节，权衡时间/精度。
- dmsign 稳定化：`dmsign_vjp` 内部 `eps` 默认 1e-3，如 A 近似亏秩可适当增大。
- 多样本：用 `--cases` 和不同 `--seed` 扩展实验覆盖面。

----------------------------------------

## 扩展方向

- 为 Newton 加线搜索/保括号，构建“全局化牛顿”。
- 统一日志为 CSV/JSON，便于批量统计绘图。
- 在训练循环中热启动：将上一次求得的 λ* 作为初值，常 1–3 步收敛。
- 缓存中间量：若在外部已有极分解产物，可缓存传入减少重复计算。

----------------------------------------

## 相关文件

- `dev/zhihu/msign.py`：msign 算子（多项式迭代）。
- `dev/zhihu/dmsign.md`：dmsign 推导与实现说明（mcsgn 块矩阵法）。
- `dev/zhihu/dmsign.py`：dmsign 实现（包含 `mcsgn` 与 `dmsign_vjp`）。
- `dev/zhihu/solve_lambda.py`：本文档对应的测试与求解脚本。

