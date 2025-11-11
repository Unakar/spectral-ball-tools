import torch
import numpy as np
from itertools import chain, islice, repeat
from typing import Optional

# ==============================
# Coefficient Sets
# ==============================

# Old Polar-Express coefficients (with deflation in original code)
_OLD_POLAR_EXPRESS_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# New Megatron (Muon) coefficients - NO deflation
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

# ==============================
# Helper Functions
# ==============================

def rms_norm(tensor: torch.Tensor) -> float:
    """Compute root-mean-square (RMS) of all elements."""
    return torch.sqrt(torch.mean(tensor ** 2)).item()

def msign_svd(g: torch.Tensor) -> torch.Tensor:
    """Exact matrix sign via SVD."""
    u, s, vh = torch.linalg.svd(g, full_matrices=False)
    return u @ torch.diag(torch.sign(s)) @ vh

# ==============================
# Old msign (with deflation)
# ==============================

def _deflate_coeffs(abc: tuple, deflation_eps: float) -> tuple:
    a, b, c = abc
    return (
        a / (1 + deflation_eps),
        b / (1 + deflation_eps) ** 3,
        c / (1 + deflation_eps) ** 5,
    )

def old_msign_kernel(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Original msign with deflation (eps=0.01)."""
    assert G.ndim >= 2
    assert steps > 0

    deflation_eps = 0.01
    X = G.bfloat16()

    transpose = G.size(-2) > G.size(-1)
    if transpose:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + deflation_eps) + 1e-7)

    # Prepare coefficients
    hs = [
        _deflate_coeffs(coeffs, deflation_eps)
        for coeffs in chain(
            islice(_OLD_POLAR_EXPRESS_COEFFS, steps),
            repeat(_OLD_POLAR_EXPRESS_COEFFS[-1], max(0, steps - len(_OLD_POLAR_EXPRESS_COEFFS))),
        )
    ]

    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transpose:
        X = X.mT

    return X.float()

# ==============================
# New Megatron (Muon) msign
# ==============================

def _muon_newton_schulz_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)  # B = b*A + c*A@A
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)  # X = a*X + B@X
    return X

def muon_msign_kernel(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str = "polar_express",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Megatron/Muon-style msign without deflation."""
    if x.ndim < 2:
        raise ValueError("x must have at least 2 dims")
    if x.dtype != torch.float32:
        x = x.float()

    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    X = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)

    coeffs = MUON_COEFFICIENT_SETS[coefficient_type]
    coeff_len = len(coeffs)
    if steps % coeff_len != 0:
        # For fairness, we allow non-multiple steps by cycling (relax constraint)
        # But original code requires multiple; here we just cycle safely.
        pass  # We'll use modulo indexing

    for i in range(steps):
        a, b, c = coeffs[i % coeff_len]
        X = _muon_newton_schulz_step(X, a, b, c)

    if transpose:
        X = X.mT
    return X

# ==============================
# Main Comparison
# ==============================

def compare_msign_methods():
    torch.manual_seed(42)
    matrix_size = (256, 1024)
    G = torch.randn(matrix_size, dtype=torch.float32)

    print(f"测试矩阵大小: {matrix_size}")
    print(f"矩阵谱范数: {torch.linalg.norm(G, ord=2).item():.6f}\n")

    # Ground truth
    print("计算 SVD 基准 (ground truth)...")
    S_svd = msign_svd(G)

    # Choose steps that work for both
    # Old: works with any steps >=1
    # New: ideally multiple of 8, so we use 8, 16
    steps_list = [8]

    print(f"{'步骤':<6} {'方法':<12} {'绝对RMS误差':<15} {'最大绝对误差':<15}")
    print("-" * 60)

    for steps in steps_list:
        # Old msign
        S_old = old_msign_kernel(G, steps)
        err_old = S_old - S_svd
        rms_old = rms_norm(err_old)
        max_old = torch.max(torch.abs(err_old)).item()

        # Megatron (Muon) msign
        S_muon = muon_msign_kernel(G, steps)
        err_muon = S_muon - S_svd
        rms_muon = rms_norm(err_muon)
        max_muon = torch.max(torch.abs(err_muon)).item()

        print(f"{steps:<6} {'Old':<12} {rms_old:<15.6e} {max_old:<15.6e}")
        print(f"{'':<6} {'Megatron':<12} {rms_muon:<15.6e} {max_muon:<15.6e}")
        print()  # blank line between step groups

    # Additional: test on multiple random matrices (16 steps)
    print("在多个随机矩阵上测试 (16步, 5次):")
    old_rms_list, muon_rms_list = [], []
    for i in range(5):
        G_test = torch.randn(matrix_size, dtype=torch.float32)
        S_true = msign_svd(G_test)
        
        S_old_test = old_msign_kernel(G_test, 8)
        S_muon_test = muon_msign_kernel(G_test, 8)
        
        rms_old = rms_norm(S_old_test - S_true)
        rms_muon = rms_norm(S_muon_test - S_true)
        
        old_rms_list.append(rms_old)
        muon_rms_list.append(rms_muon)
        
        print(f"测试 {i+1:>2}: Old={rms_old:.6e}, Megatron={rms_muon:.6e}")

    print(f"\n平均 RMS 误差 (8步, 5次):")
    print(f"Old      : {np.mean(old_rms_list):.6e} ± {np.std(old_rms_list):.6e}")
    print(f"Megatron : {np.mean(muon_rms_list):.6e} ± {np.std(muon_rms_list):.6e}")

if __name__ == "__main__":
    compare_msign_methods()