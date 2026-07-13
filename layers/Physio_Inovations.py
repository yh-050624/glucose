import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    用于自动抑制噪声通道 (针对 Pt563 乱序噪声)
    """

    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class PhysioFeatureEnhancer(nn.Module):
    """
    [Innovation I] Robust Morpho-Kinetic Enhancer
    位置: Feature Space
    作用: 对原始生理信号进行去噪和形态提取。
    """

    def __init__(self, c_in, seq_len, d_model, dropout=0.1):
        super(PhysioFeatureEnhancer, self).__init__()

        # 1. Glucose Stream (CNN + SE + GroupNorm)
        # 提取波形形态，SEBlock 负责压制噪声
        self.glucose_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ELU(),
            SEBlock(16),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.GroupNorm(1, 1),
            nn.ELU()
        )
        self.glucose_proj = nn.Linear(seq_len, d_model)

        # 2. Intervention Stream (Bottleneck MLP)
        # 提取强度信息
        d_hidden = d_model // 4
        self.intervention_mlp = nn.Sequential(
            nn.Linear(seq_len, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model)
        )

        self.dropout = nn.Dropout(dropout)

        # ReZero Init: 确保初始零输出
        self.alpha = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        # x: [B, L, N]
        inter_data = x[:, :, :-1].permute(0, 2, 1)
        inter_emb = self.intervention_mlp(inter_data)

        glu_data = x[:, :, -1:].permute(0, 2, 1)
        glu_feat = self.glucose_conv(glu_data)
        glu_emb = self.glucose_proj(glu_feat)

        out = torch.cat([inter_emb, glu_emb], dim=1)
        return self.alpha * self.dropout(out)


class CausalMultiScaleAligner(nn.Module):
    """
    [Innovation II] Temporal-Space Causal Aligner

    关键修正:
    卷积必须作用于 [Time] 维度，而不是 [Variate] 维度！

    Input: x_enc [B, L, N]
    Process: Conv1d along L
    Output: Projected to [B, N, D]
    """

    def __init__(self, n_vars, seq_len, d_model, dropout=0.1):
        super(CausalMultiScaleAligner, self).__init__()
        self.n_vars = n_vars

        # 多尺度时间卷积: 捕捉 1min, 3min, 5min 的滞后
        # groups=n_vars 保证每个变量独立处理时间，互不干扰
        self.conv_short = nn.Conv1d(n_vars, n_vars, kernel_size=1, groups=n_vars)
        self.conv_medium = nn.Conv1d(n_vars, n_vars, kernel_size=3, padding=2, groups=n_vars)
        self.conv_long = nn.Conv1d(n_vars, n_vars, kernel_size=5, padding=4, groups=n_vars)

        # 将时间特征 [L] 投影到 Embedding 维度 [D]
        self.projector = nn.Linear(seq_len * 3, d_model)

        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        # x: [B, L, N] -> [B, N, L]
        # 这样 Conv1d 就在 L (时间) 上滑动了
        x_in = x.permute(0, 2, 1)

        # Temporal Convolution
        feat1 = self.conv_short(x_in)

        # Causal Crop for Medium (kernel 3, pad 2 -> remove last 2)
        feat2 = self.conv_medium(x_in)
        feat2 = feat2[:, :, :-2]

        # Causal Crop for Long (kernel 5, pad 4 -> remove last 4)
        feat3 = self.conv_long(x_in)
        feat3 = feat3[:, :, :-4]

        # Concatenate Multi-scale Features: [B, N, L*3]
        concat = torch.cat([feat1, feat2, feat3], dim=-1)

        # Project to Feature Space: [B, N, D]
        out = self.projector(concat)

        return self.alpha * self.dropout(out)


class PhysioAdaptiveAttention(nn.Module):
    """
    [Innovation III] Physio-Adaptive Attention (PAA)

    改进:
    移除了无效的 Local Conv (变量间卷积)。
    改为纯粹的 Self-Attention + 可学习的生理先验 Bias。
    """

    def __init__(self, n_vars, d_model, n_heads=4, dropout=0.1):
        super(PhysioAdaptiveAttention, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 生理先验矩阵 (Physio Bias)
        # 初始化为关注 Target (最后一列)
        self.prior_bias = nn.Parameter(torch.randn(n_vars, n_vars) * 0.01)
        self.gate = nn.Parameter(torch.zeros(1))

        with torch.no_grad():
            self.prior_bias.fill_(0.0)
            self.prior_bias[-1, :] = 0.5  # 让所有变量关注血糖

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.alpha = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        B, N, D = x.shape

        Q = self.W_Q(x).reshape(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.W_K(x).reshape(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.W_V(x).reshape(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_head ** 0.5)

        # Add Gated Prior
        # Scores: [B, H, N, N]
        # Prior: [1, 1, N, N]
        prior = torch.sigmoid(self.gate) * self.prior_bias.unsqueeze(0).unsqueeze(0)
        scores = scores + prior

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V).permute(0, 2, 1, 3).reshape(B, N, D)

        out = self.out_proj(out)
        return self.alpha * self.norm(self.dropout(out))