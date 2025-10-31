# Spectral Ball 工具箱

本工具箱用于求解和评估一类标量方程的数值策略，该方程源于**带约束的谱范数矩阵优化问题**：

$$
f(\lambda) = \langle \Theta,\ \mathrm{msign}(G + \lambda \Theta) \rangle = 0,
$$

其中：
- $G \in \mathbb{R}^{m \times n}$ 是给定的数据矩阵，
- $\Theta = u_1 v_1^\top$ 是一个**秩-1 约束矩阵**（通常来自某个先前问题的主奇异向量），
- $\mathrm{msign}(\cdot)$ 是**矩阵符号函数**，定义为极分解中的酉因子：  
  $\mathrm{msign}(A) = A (A^\top A)^{-1/2}$（当 $A$ 满秩时），满足 $\|\mathrm{msign}(A)\|_2 = 1$，
- $\langle X, Y \rangle = \mathrm{tr}(X^\top Y)$ 表示 Frobenius 内积。

---

## 🎯 问题来源与动机

该方程出现在如下带约束优化问题中：

$$
\max_{\|\Phi\|_2 = 1} \langle G, \Phi \rangle \quad \text{满足} \quad \langle \Theta, \Phi \rangle = 0.
$$

利用拉格朗日对偶性及恒等式 $\max_{\|\Phi\|_2 \le 1} \langle A, \Phi \rangle = \|A\|_*$（核范数），最优解可表示为 $\Phi(\lambda) = \mathrm{msign}(G + \lambda \Theta)$。正交约束 $\langle \Theta, \Phi \rangle = 0$ 随之转化为求解标量方程 $f(\lambda) = 0$。

因此，**高效且稳定地求解 $f(\lambda) = 0$ 是此类问题的核心计算瓶颈**——常见于鲁棒 PCA、对抗训练和谱正则化等场景。

---

## 🔧 实现的算法

所有方法均通过调用 $\mathrm{msign}(\cdot)$（以及可选的其导数）来求解 $f(\lambda) = 0$。各方法特点如下：

| 方法       | 类型         | 是否需要导数？ | 收敛速度   | 鲁棒性     |
|-----------|--------------|----------------|------------|------------|
| **Brent** | 区间括号法    | ❌             | 超线性      | ✅ 高（若成功括号根） |
| **Secant**| 局部法        | ❌             | 超线性      | ⚠️ 中等（依赖初值） |
| **Fixed** | 不动点迭代    | ❌             | 线性        | ⚠️ 低（启发式阻尼） |
| **Newton**| 局部法        | ✅（通过 `dmsign`）| 二次收敛   | ✅ 高（在根附近） |

### 算法详解

- **Brent 方法**：结合二分法（安全）、割线法和逆二次插值。需初始区间 $[a,b]$ 满足 $f(a)f(b) < 0$。我们从 $\lambda=0$ 出发自动向外搜索以找到这样的区间。
  
- **割线法（Secant）**：通过有限差分近似导数，作为无导数的 Newton 替代方案。

- **不动点法（Fixed）**：受文献中“公式 (10)”启发，更新规则为 $\lambda \leftarrow \lambda - c(\lambda) f(\lambda)$，其中 $c(\lambda)$ 利用核范数信息近似 $1/f'(\lambda)$。包含回溯机制以抑制震荡——但**不保证收敛**。

- **牛顿法（Newton）**：使用精确导数  
  $$
  f'(\lambda) = \langle \Theta,\ D\mathrm{msign}_{A}[\Theta] \rangle,
  $$  
  其中 $A = G + \lambda \Theta$，$D\mathrm{msign}_A[H]$ 是 `msign` 在 $A$ 处沿方向 $H$ 的 Fréchet 导数。实现方式包括：
  - **`dmsign_vjp`**：基于**mcsgn 块矩阵技巧**的向量-Jacobian 积（VJP）形式（避免显式 Jacobian），
  - 若 `dmsign` 不可用，则回退到有限差分。

---

## 📊 输出与 CSV 字段说明

基准测试脚本（位于 `scripts/`）生成的 CSV 文件包含以下字段：

| 字段 | 含义 |
|------|------|
| `m`, `n` | 矩阵维度 |
| `method` | 使用的求解器（`brent`、`secant`、`fixed`、`newton`） |
| `seed` | 随机种子（用于复现） |
| `iters` | 求解器迭代次数 |
| `f_evals` | `msign` 调用次数（即 $f(\lambda)$ 的求值次数） |
| `time_ms` | 墙钟时间（毫秒） |
| `f_abs` | 终止时 $|f(\lambda)|$ 的绝对值 |
| `converged` | 布尔值：是否达到容差？ |
| `f0` | 初始残差 $f(0)$ |
| `rel_phi_err0` | `msign(G)` 相对于高精度极分解的相对误差 |
| `rel_phi_err_lam` | 在最终 $\lambda$ 处的相同误差 |

> 💡 **注意**：`rel_phi_err` 用于评估你的 `msign` 实现（如 10 步多项式迭代）相对于 SVD 极分解参考解的精度。

---

## 🚀 快速示例

使用 Brent 方法求解（自动括号）：

```bash
python solve_lambda.py --method brent --m 64 --n 32 --seed 0
```

对比所有方法：

```bash
python solve_lambda.py --method all --m 64 --n 32 --seed 0
```

使用 Newton 法（需 `dmsign.py`）：

```bash
python solve_lambda.py --method newton --m 64 --n 32
```

运行网格基准测试：

```bash
bash scripts/bench_grid.sh results/
# 输出：results/bench_all.csv
```

---

## 📁 关键文件

- `msign.py`：高效的多项式迭代实现 $\mathrm{msign}(A)$（Polar Express 风格）。
- `dmsign.py`：基于 **mcsgn 块矩阵法** 的 `msign` Fréchet 导数（支持 Newton 法）。
- `solve_lambda.py`：包含全部四种算法的命令行求解器。
- `scripts/`：基准测试脚本（`bench.sh`、`bench_grid.sh`）。


---

