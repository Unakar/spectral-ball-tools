## Spectral Ball Tools

求解与评估如下标量方程：

  f(λ) = <Θ, msign(G + λ Θ)> = 0

这里 msign(·) 是矩阵的核函数。本工具箱包含：

- 高效 `msign`（多项式迭代，Polar Express 风格）
- `dmsign` 的 VJP 实现（mcsgn 块矩阵法，无需 SVD/EVD）
- 多种一维根求解算法与对比脚本

适合在训练循环或研究场景中评估不同求解策略的收敛时间与数值精度。

----------------------------------------

## 目录

- `msign.py`：矩阵 msign 算子。
- `dmsign.py`：dmsign 的 VJP（伴随）实现与 `mcsgn`。
- `solve_lambda.py`：求解 f(λ)=0 的 CLI 脚本（Brent / Newton / Fixed / Secant）。
- `scripts/`：批量对比与网格基准脚本（生成 CSV）。
- 上游原理文档：`../dmsign.md`、`../instruction.md`（保持在上级目录）。

----------------------------------------

## 快速开始

依赖：Python 3.9+，PyTorch（CPU/GPU 皆可）。

示例（Brent，64×32）：

```
python3 solve_lambda.py --method brent --m 64 --n 32 --seed 0 --device auto
```

跑全部方法：

```
python3 solve_lambda.py --method all --m 64 --n 32 --seed 0
```

固定点法：

```
python3 solve_lambda.py --method fixed --m 64 --n 32 --msign-steps 10 --tol 1e-8
```

Newton（使用 dmsign 的 VJP 计算导数）：

```
python3 solve_lambda.py --method newton --m 64 --n 32
```

输出会包含迭代数、函数评估次数、耗时（毫秒）与收敛标志，并对 msign 的单位因子与参考极分解进行相对误差评估。

----------------------------------------

## 算法一览

- Brent（brent）：带括号的全局稳健法，融合二分/割线/逆二次插值。
- Fixed（fixed/eq10）：预条件残差迭代 λ←λ−c(λ)f(λ)，带回溯抑制震荡。
- Secant（secant）：无导数局部法，作为对比/回退。
- Newton（newton）：f'(λ)=<Θ, Dmsign_A[Θ]>，导数用 dmsign VJP（mcsgn 块矩阵法）。


----------------------------------------

## 批量对比脚本（CSV 输出）

脚本位于 `scripts/` 下，默认不会写入仓库外路径。

- 单尺寸、遍历方法与多种种子：

```
bash scripts/bench.sh 64 32 "brent secant fixed newton" "0 1 2 3" out_64x32.csv
```

环境变量可调整：`MSIGN_STEPS`、`TOL`、`DEVICE`、`CASES`。

- 尺寸网格基准，并合并 CSV：

```
bash scripts/bench_grid.sh benchmarks
```

将生成 `benchmarks/bench_*.csv` 与合并后的 `benchmarks/bench_all.csv`。

CSV 字段：

```
m,n,method,seed,iters,f_evals,time_ms,f_abs,converged,f0,rel_phi_err0,rel_phi_err_lam
```


