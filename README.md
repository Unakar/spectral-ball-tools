# Spectral Ball Tools

[中文文档 | Chinese Version](README_zh.md)

This toolbox solves and benchmarks numerical strategies for the scalar root-finding problem that arises in **constrained spectral-norm matrix optimization**:

$$
f(\lambda) = \langle \Theta,\ \mathrm{msign}(G + \lambda \Theta) \rangle = 0,
$$

where:
- $G \in \mathbb{R}^{m \times n}$ is a given data matrix,
- $\Theta = u_1 v_1^\top$ is a **rank-1 constraint matrix** (typically from the leading singular vectors of a prior problem),
- $\mathrm{msign}(\cdot)$ is the **matrix sign function**, defined as the polar factor:  
  $\mathrm{msign}(A) = A (A^\top A)^{-1/2}$ (for full-rank $A$), which satisfies $\|\mathrm{msign}(A)\|_2 = 1$,
- $\langle X, Y \rangle = \mathrm{tr}(X^\top Y)$ is the Frobenius inner product.

---

## 🎯 Why This Problem?

This equation emerges when solving the constrained optimization problem:

$$
\max_{\|\Phi\|_2 = 1} \langle G, \Phi \rangle \quad \text{subject to} \quad \langle \Theta, \Phi \rangle = 0.
$$

Using Lagrange duality and the fact that $\max_{\|\Phi\|_2 \le 1} \langle A, \Phi \rangle = \|A\|_*$ (nuclear norm), the optimal $\Phi$ takes the form $\Phi(\lambda) = \mathrm{msign}(G + \lambda \Theta)$. The orthogonality constraint $\langle \Theta, \Phi \rangle = 0$ then reduces to solving $f(\lambda) = 0$ for a scalar $\lambda$.

Thus, **efficiently and robustly solving $f(\lambda) = 0$ is the key computational bottleneck** in this class of problems—common in robust PCA, adversarial training, and spectral regularization.

---

## 🔧 Algorithms Implemented

All methods solve $f(\lambda) = 0$ using only evaluations of $\mathrm{msign}(\cdot)$ (and optionally its derivative). Each has distinct trade-offs:

| Method     | Type          | Derivative Required? | Convergence | Robustness |
|-----------|---------------|----------------------|-------------|------------|
| **Brent** | Bracketing    | ❌                   | Superlinear | ✅ High (guaranteed if bracketed) |
| **Secant**| Local         | ❌                   | Superlinear | ⚠️ Moderate (needs good init) |
| **Fixed** | Fixed-point   | ❌                   | Linear      | ⚠️ Low (heuristic damping) |
| **Newton**| Local         | ✅ (via `dmsign`)    | Quadratic   | ✅ High (near root) |

### Algorithm Details

- **Brent’s method**: Combines bisection (safe), secant, and inverse quadratic interpolation. Requires an initial interval $[a,b]$ where $f(a)f(b) < 0$. We automatically search for such a bracket by expanding outward from $\lambda=0$.
  
- **Secant method**: Uses finite differences of $f(\lambda)$ to approximate the derivative. Serves as a derivative-free local alternative to Newton.

- **Fixed-point iteration**: Inspired by the algebraic manipulation in "Formula (10)" from the literature. Updates $\lambda \leftarrow \lambda - c(\lambda) f(\lambda)$, where $c(\lambda)$ approximates $1/f'(\lambda)$ using nuclear norm information. Includes backtracking to reduce oscillation—but **not guaranteed to converge**.

- **Newton’s method**: Uses the exact derivative  
  $$
  f'(\lambda) = \langle \Theta,\ D\mathrm{msign}_{A}[\Theta] \rangle,
  $$  
  where $A = G + \lambda \Theta$ and $D\mathrm{msign}_A[H]$ is the Fréchet derivative of `msign` at $A$ in direction $H$. Implemented via:
  - **`dmsign_vjp`**: A vector-Jacobian product (VJP) formulation using the **mcsgn block matrix trick** (avoids explicit Jacobian),
  - Falls back to finite differences if `dmsign` is unavailable.

---

## 📊 Output & CSV Fields

Benchmark scripts (in `scripts/`) generate CSV files with the following columns:

| Field | Description |
|------|-------------|
| `m`, `n` | Matrix dimensions |
| `method` | Solver used (`brent`, `secant`, `fixed`, `newton`) |
| `seed` | Random seed for reproducibility |
| `iters` | Number of solver iterations |
| `f_evals` | Number of calls to `msign` (i.e., evaluations of $f(\lambda)$) |
| `time_ms` | Wall-clock time in milliseconds |
| `f_abs` | Absolute value of $f(\lambda)$ at termination |
| `converged` | Boolean: did the solver meet tolerance? |
| `f0` | Initial residual $f(0)$ |
| `rel_phi_err0` | Relative error of `msign(G)` vs. high-precision polar decomposition |
| `rel_phi_err_lam` | Same, but at the final $\lambda$ |

> 💡 **Note**: `rel_phi_err` measures the accuracy of your `msign` implementation (e.g., 10-step polynomial iteration) against a reference SVD-based polar factor.

---

## 🚀 Quick Examples

Solve with Brent (auto-bracketing):

```bash
python solve_lambda.py --method brent --m 64 --n 32 --seed 0
```

Compare all methods:

```bash
python solve_lambda.py --method all --m 64 --n 32 --seed 0
```

Run Newton with analytic derivative (requires `dmsign.py`):

```bash
python solve_lambda.py --method newton --m 64 --n 32
```

Run grid benchmark:

```bash
bash scripts/bench_grid.sh results/
# Outputs: results/bench_all.csv
```

---

## 📁 Key Files

- `msign.py`: Efficient polynomial-iteration implementation of $\mathrm{msign}(A)$ (Polar Express style).
- `dmsign.py`: Fréchet derivative of `msign` via the **mcsgn block matrix method** (enables Newton).
- `solve_lambda.py`: CLI solver with all four algorithms.
- `scripts/`: Benchmarking utilities (`bench.sh`, `bench_grid.sh`).


---

