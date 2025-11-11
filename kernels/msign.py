import torch
from typing import Optional


def _muon_newton_schulz_step(
    X: torch.Tensor, a: float, b: float, c: float
) -> torch.Tensor:
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X


@torch.compile
def msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    if G.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    if G.dtype != torch.float32:
        G = G.float()

    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G

    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)

    coeffs = [
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
    ]

    for i in range(steps):
        a, b, c = coeffs[i % 8]
        X = _muon_newton_schulz_step(X, a, b, c)

    if transpose:
        X = X.mT

    return X
