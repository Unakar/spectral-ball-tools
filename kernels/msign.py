import torch

# 预计算的多项式系数 (a, b, c)，用于每一步迭代
ABC_LIST: list[tuple[float, float, float]] = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# 对前 N-1 项应用安全缩放（提升数值稳定性），最后一项保持不变
ABC_LIST_STABLE: list[tuple[float, float, float]] = [
    (a / 1.01, b / (1.01**3), c / (1.01**5)) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]


@torch.no_grad()
def msign(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    使用 Polar Express 算法计算矩阵符号函数（matrix sign function）。
    论文：https://arxiv.org/abs/2505.16932

    功能：近似 sign(G) = G (G^T G)^{-1/2}，用于极分解。
    """
    assert G.ndim >= 2, "Input tensor must have at least 2 dimensions."

    # 如果行数 > 列数，转置以提升效率（处理瘦矩阵）
    should_transpose = G.size(-2) > G.size(-1)
    x = G.bfloat16()  # 使用 bfloat16 节省内存并加速
    if should_transpose:
        x = x.mT  # 批量转置

    # 归一化：防止数值溢出，乘以 1.01 作为安全裕度
    norm = x.norm(dim=(-2, -1), keepdim=True)
    x = x / (norm * 1.01)

    # 迭代更新：使用预计算的多项式系数
    for step in range(steps):
        # 如果步数超出预设系数数量，复用最后一个系数
        if step < len(ABC_LIST_STABLE):
            a, b, c = ABC_LIST_STABLE[step]
        else:
            a, b, c = ABC_LIST_STABLE[-1]

        # 计算 S = X X^T （对称 Gram 矩阵）
        S = x @ x.mT

        # 按照公式：X_{new} = (a I + b S + c S^2) X
        # 为避免显式构造 I，我们直接操作对角线

        # 先计算 c * S
        Y = c * S

        # 在 Y 的对角线上加上 b → 相当于 b I + c S
        Y.diagonal(dim1=-2, dim2=-1).add_(b)

        # 再乘以 S → (b I + c S) S = b S + c S^2
        Y = Y @ S

        # 在结果的对角线上加上 a → a I + b S + c S^2
        Y.diagonal(dim1=-2, dim2=-1).add_(a)

        # 最后乘以原始 X：X_new = (a I + b S + c S^2) X
        x = Y @ x

    # 如果之前转置过，再转回来
    if should_transpose:
        x = x.mT

    # 替换 NaN/Inf 为 0（数值安全兜底）
    x = torch.nan_to_num(x)

    # 返回 float32（兼容下游）
    return x.float()