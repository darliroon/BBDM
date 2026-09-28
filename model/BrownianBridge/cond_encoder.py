import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageContextEncoder(nn.Module):
    """条件图 C 编码器: 图像 -> token 序列, 供 UNet cross-attention 使用。

    - 输入 (B, 3, H, W) 归一化图像, 输出 (B, token_grid^2, context_dim)
    - token 空间位置与输入图对齐 (A 从 C 提取, 同坐标系, UNet 可按位置查询 C)
    - 输出投影与位置嵌入零初始化: 训练初期 C 的贡献≈0,
      不扰动 A->B 近域桥, C 的影响随训练渐进长出 (ControlNet 式引入)
    - null_embedding: 条件丢弃 (CFG 训练) 与无条件采样时的可学习占位 token
    """

    def __init__(self, in_channels=3, width=128, context_dim=512, token_grid=32, num_blocks=3):
        super().__init__()
        self.token_grid = token_grid
        self.context_dim = context_dim

        blocks = [nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 1, 1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )]
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(width, width, 3, 2, 1),
                nn.GroupNorm(8, width),
                nn.SiLU(),
            ))
        self.blocks = nn.ModuleList(blocks)

        self.proj = nn.Linear(width, context_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, token_grid * token_grid, context_dim))
        self.null_embedding = nn.Parameter(torch.randn(1, 1, context_dim) * 0.02)

        self.reset_zero_init()

    def reset_zero_init(self):
        """恢复零初始化 (weights_init 会覆盖普通参数, 这里夺回零初始化)"""
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.pos_embedding)

    def forward(self, c, drop_prob=0.0):
        """
        :param c: (B, 3, H, W) 条件图
        :param drop_prob: 整样本条件丢弃概率 (仅训练期 >0)
        :return: (B, token_grid^2, context_dim)
        """
        f = c
        for block in self.blocks:
            f = block(f)
        f = F.adaptive_avg_pool2d(f, (self.token_grid, self.token_grid))
        tokens = f.flatten(2).transpose(1, 2)              # (B, N, width)
        tokens = self.proj(tokens) + self.pos_embedding    # (B, N, context_dim)

        if drop_prob > 0.0:
            keep = torch.rand(c.shape[0], device=c.device) >= drop_prob
            tokens = torch.where(keep.view(-1, 1, 1), tokens,
                                 self.null_embedding.to(tokens.dtype))
        return tokens

    def null_tokens(self, batch_size):
        """无条件 (C 缺失 / CFG 分支) 的 null token 序列"""
        n = self.token_grid * self.token_grid
        return self.null_embedding.expand(batch_size, n, self.context_dim)
