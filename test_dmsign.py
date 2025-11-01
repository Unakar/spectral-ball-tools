import torch
import torch.nn.functional as F
from msign import msign as msign_original  # 原始无梯度版本
from dmsign import dmsign

# -----------------------------
# 1. 构造一个支持梯度的 msign（用于 autograd）
# -----------------------------
def msign_autograd(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    支持自动微分的 msign 实现（仅用于测试！实际训练请用原始版本 + dmsign 反传）
    """
    assert M.ndim >= 2
    should_transpose = M.size(-2) > M.size(-1)
    x = M.to(torch.bfloat16)
    if should_transpose:
        x = x.mT

    # 归一化（与原始 msign 一致）
    norm = x.norm(dim=(-2, -1), keepdim=True) * 1.01
    x = x / norm.clamp_min(1e-12)

    # 使用原始系数（注意：这里不能用 ABC_LIST_STABLE，因为那是为 no_grad 设计的）
    # 为简化，我们直接复制原始 msign 的系数（未缩放版）
    ABC_LIST = [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
        (1.875, -1.25, 0.375),
    ]
    ABC_LIST_STABLE = [
        (a / 1.01, b / (1.01**3), c / (1.01**5)) for (a, b, c) in ABC_LIST[:-1]
    ] + [ABC_LIST[-1]]

    for step in range(steps):
        a, b, c = ABC_LIST_STABLE[step] if step < len(ABC_LIST_STABLE) else ABC_LIST_STABLE[-1]
        s = x @ x.mT
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    if should_transpose:
        x = x.mT
    x = torch.nan_to_num(x)
    return x.float()


# -----------------------------
# 2. 测试函数
# -----------------------------
def test_dmsign_correctness():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 测试多种形状（方阵 + 矩形）
    shapes = [(32,32),(64,256)]

    for shape in shapes:
        print(f"\nTesting shape: {shape}")
        m, n = shape

        # 随机输入矩阵（避免奇异）
        A = torch.randn(m, n, device=device, dtype=torch.float32, requires_grad=True)

        # 前向：计算 O = msign(A)
        O = msign_autograd(A, steps=5)

        # 构造一个随机上游梯度（模拟 loss 对 O 的梯度）
        C = torch.randn_like(O)

        # 计算 loss = sum(C * O)，则 dL/dO = C
        loss = (C * O).sum()

        # 自动微分：计算 dL/dA
        loss.backward()
        grad_autograd = A.grad.clone()

        # 重置梯度
        A.grad = None

        # 使用 dmsign 计算解析梯度：dA = dmsign(A, C)
        grad_dmsign = dmsign(A.detach(), C, msign_fn=msign_original, steps=5, eps=1e-3)

        # 比较
        diff = torch.abs(grad_autograd - grad_dmsign)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        # 判断是否一致（容忍 bfloat16 和迭代近似误差）
        assert torch.allclose(grad_autograd, grad_dmsign, atol=1e-3, rtol=1e-1), \
            f"Gradient mismatch for shape {shape}! Max diff = {max_diff}"

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_dmsign_correctness()