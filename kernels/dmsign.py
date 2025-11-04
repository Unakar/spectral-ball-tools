import torch
from typing import Optional, Callable
from kernels.msign import msign  # 导入已实现的高效矩阵符号函数

 # See https://kexue.fm/archives/11025

@torch.no_grad()
def mcsgn(M: torch.Tensor, steps: int = 5, eps: float = 1e-20) -> torch.Tensor:
    """
    对方阵 M 计算其矩阵符号函数（matrix sign function）。
    
    与 msign.msign 使用相同的多项式迭代算法，但采用 Frobenius 范数（通过 trace 归一化）
    进行缩放——这是为了与 dmsign 中的块矩阵技巧保持一致。
    """
    # 确保输入是二维方阵
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "mcsgn 要求输入为二维方阵"

    # 计算 Frobenius 范数的平方：||M||_F^2 = trace(M^T M)
    fro_norm_sq = torch.trace(M @ M).clamp_min(eps)  # 防止除零，加极小值保护
    
    # 将 M 归一化为单位 Frobenius 范数
    X = M / torch.sqrt(fro_norm_sq)

    # 调用通用的 msign 函数（它支持批量输入，因此临时增加一个 batch 维度）
    O = msign(X.unsqueeze(0), steps=steps).squeeze(0)  # 计算后去掉 batch 维度

    # 将 NaN 或无穷大替换为 0，确保数值稳定性
    return torch.nan_to_num(O)


@torch.no_grad()
def dmsign(
    A: torch.Tensor,
    C: torch.Tensor,
    msign_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    steps: int = 6,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    计算矩阵符号函数的向量-Jacobian 积（VJP），即反向传播梯度。

    已知损失函数对 O = msign(A) 的梯度为 C，
    本函数计算损失对原始矩阵 A 的梯度 dA，满足：
        ⟨C, D msign_A[H]⟩ = ⟨dA, H⟩   （对任意扰动 H 成立）

    实现基于 dmsign.md 中的“块矩阵技巧”：
      1. 构造 Sylvester 方程：A1·X + X·B1 = C
      2. 通过一个更大的块矩阵的矩阵符号函数间接求解该方程
      3. 最终梯度 dA = X - O·Xᵀ·O
    """
    # 输入校验：A 和 C 都必须是二维张量，且形状一致
    assert A.ndim == 2 and C.ndim == 2
    m, n = A.shape
    assert C.shape == (m, n), "梯度 C 的形状必须与 A 一致"

    # 必须提供 msign 函数（例如来自 msign.py 的实现）
    if msign_fn is None:
        raise ValueError("dmsign 需要传入 msign_fn，例如 msign.msign")

    # 第一步：计算 A 的极分解中的正交因子 O = msign(A)
    O = msign_fn(A)  # shape: (m, n)

    # 第二步：构造 Sylvester 方程所需的两个对称块
    A1 = A @ O.T      # shape: (m, m)，近似为对称正定
    B1 = O.T @ A      # shape: (n, n)，同上

    # 第三步：对 A1 和 B1 进行对称化和正则化（提升数值稳定性）
    I_m = torch.eye(m, dtype=A.dtype, device=A.device)  # m×m 单位矩阵
    I_n = torch.eye(n, dtype=A.dtype, device=A.device)  # n×n 单位矩阵
    
    A1 = 0.5 * (A1 + A1.T) + eps * I_m  # 强制对称，并加上小扰动防止奇异
    B1 = 0.5 * (B1 + B1.T) + eps * I_n

    # 第四步：构造 (m+n)×(m+n) 的块矩阵 M：
    #     M = [ A1     -C  ]
    #         [  0    -B1  ]
    M = torch.zeros(m + n, m + n, dtype=A.dtype, device=A.device)
    M[:m, :m] = A1          # 左上块：A1
    M[:m, m:] = -C          # 右上块：-C
    M[m:, m:] = -B1         # 右下块：-B1（左下块保持为 0）

    # 第五步：对整个块矩阵 M 应用矩阵符号函数
    S = mcsgn(M, steps=max(3, steps))  # 至少使用 3 步保证收敛

    # 第六步：从结果 S 中提取 Sylvester 方程的解 X
    # 根据理论，X = -0.5 × S 的右上子块
    X = -0.5 * S[:m, m:]  # shape: (m, n)

    # 第七步：将 X 投影到 Stiefel 流形（正交约束）的切空间上，
    # 得到最终关于 A 的梯度
    dA = X - O @ X.T @ O  # shape: (m, n)

    # 替换可能的 NaN/Inf，确保输出干净
    return torch.nan_to_num(dA)