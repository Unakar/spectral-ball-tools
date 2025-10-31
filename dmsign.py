import torch
from typing import Tuple, Optional, Callable


@torch.no_grad()
def mcsgn(M: torch.Tensor, steps: int = 6, eps: float = 1e-20) -> torch.Tensor:
    """
    Matrix sign for square matrices via stabilized 5th-order polynomial iteration.
    Uses coefficients consistent with the reference in dmsign.md (already scaled).
    """
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "mcsgn requires a square 2D matrix"
    # Normalize for numerical stability
    tr_m2 = torch.trace(M @ M).clamp_min(eps)
    X = M / torch.sqrt(tr_m2)

    ABC = [
        (8.287212018145622, -23.59588651909882, 17.300387312530923),
        (4.107059111542197,  -2.9478499167379084, 0.54484310829266),
        (3.9486908534822938, -2.908902115962947,  0.5518191394370131),
        (3.3184196573706055, -2.488488024314878,  0.5100489401237208),
        (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
        (1.8750000000000000, -1.2500000000000000, 0.3750000000000000),
    ]
    steps = min(steps, len(ABC))
    for t in range(steps):
        a, b, c = ABC[t]
        X2 = X @ X
        X3 = X2 @ X
        X5 = X3 @ X2
        X = a * X + b * X3 + c * X5
    return torch.nan_to_num(X)


@torch.no_grad()
def dmsign_vjp(
    A: torch.Tensor,
    C: torch.Tensor,
    msign_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    steps: int = 6,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Adjoint (VJP) of D msign at A applied to C, i.e., returns dA such that
        <C, D msign_A[H]> = <dA, H> for all H.

    Implements dmsign.md block-matrix mcsgn trick to solve the Sylvester equation:
        Let O = msign(A), A1 = A O^T (m x m), B1 = O^T A (n x n),
        Solve A1 X + X B1 = C via block matrix
            M = [[A1 + eps I, -C], [0, -(B1 + eps I)]],  S = mcsgn(M),  X = -0.5 S[:m, m:]
        Then dA = X - O X^T O.
    """
    assert A.ndim == 2 and C.ndim == 2
    m, n = A.shape
    assert C.shape == (m, n)

    # Compute polar unitary factor using provided msign
    if msign_fn is None:
        raise ValueError("dmsign_vjp requires msign_fn to avoid SVD/EVD.")
    O = msign_fn(A)

    A1 = A @ O.transpose(-2, -1)  # (m, m)
    B1 = O.transpose(-2, -1) @ A  # (n, n)

    # Regularize and symmetrize diagonal blocks for numerical stability
    I_m = torch.eye(m, dtype=A.dtype, device=A.device)
    I_n = torch.eye(n, dtype=A.dtype, device=A.device)
    A1 = 0.5 * (A1 + A1.transpose(-2, -1)) + eps * I_m
    B1 = 0.5 * (B1 + B1.transpose(-2, -1)) + eps * I_n

    # Build block matrix for mcsgn
    block = torch.zeros(m + n, m + n, dtype=A.dtype, device=A.device)
    block[:m, :m] = A1
    block[:m, m:] = -C
    block[m:, m:] = -B1

    S = mcsgn(block, steps=max(3, steps))
    X = -0.5 * S[:m, m:]

    dA = X - O @ X.transpose(-2, -1) @ O
    return torch.nan_to_num(dA)


# Backward-compatible alias: compute JVP via VJP inner-product if needed.
@torch.no_grad()
def dmsign(A: torch.Tensor, H: torch.Tensor, msign_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
           steps: int = 6, eps: float = 1e-3) -> torch.Tensor:
    """
    Convenience JVP using VJP identity:
        For any H, define C = H. Then D msign_A[H] is not produced directly here;
        however, for Newton's scalar derivative f'(λ) = <Θ, D msign_A[Θ]>, one can compute
        f'(λ) = <dA, Θ> where dA = dmsign_vjp(A, C=Θ).

    This function returns a dummy zero to discourage misuse; prefer calling dmsign_vjp
    to obtain the scalar derivative via inner-product.
    """
    raise NotImplementedError(
        "Use dmsign_vjp(A, C, msign_fn) to form scalar derivatives; JVP is not explicitly constructed here."
    )


    
