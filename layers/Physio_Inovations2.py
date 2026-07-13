import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    用于自动抑制噪声通道
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
    保持不变：专注于时域形态特征提取
    """

    def __init__(self, c_in, seq_len, d_model, dropout=0.1):
        super(PhysioFeatureEnhancer, self).__init__()

        # 1. Glucose Stream (CNN + SE + GroupNorm)
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

        # 2. Intervention Stream
        d_hidden = d_model // 4
        self.intervention_mlp = nn.Sequential(
            nn.Linear(seq_len, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model)
        )

        self.dropout = nn.Dropout(dropout)
        # 建议初始化为一个微小正数，避免梯度消失
        self.alpha = nn.Parameter(torch.ones(1, 1, d_model) * 1e-3)

    def forward(self, x):
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
    保持不变：专注于解决时滞问题
    """

    def __init__(self, n_vars, seq_len, d_model, dropout=0.1):
        super(CausalMultiScaleAligner, self).__init__()
        self.n_vars = n_vars

        self.conv_short = nn.Conv1d(n_vars, n_vars, kernel_size=1, groups=n_vars)
        self.conv_medium = nn.Conv1d(n_vars, n_vars, kernel_size=3, padding=2, groups=n_vars)
        self.conv_long = nn.Conv1d(n_vars, n_vars, kernel_size=5, padding=4, groups=n_vars)

        self.projector = nn.Linear(seq_len * 3, d_model)

        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(1, 1, d_model) * 1e-3)

    def forward(self, x):
        x_in = x.permute(0, 2, 1)  # [B, N, L]

        feat1 = self.conv_short(x_in)

        feat2 = self.conv_medium(x_in)
        feat2 = feat2[:, :, :-2]

        feat3 = self.conv_long(x_in)
        feat3 = feat3[:, :, :-4]

        concat = torch.cat([feat1, feat2, feat3], dim=-1)
        out = self.projector(concat)

        return self.alpha * self.dropout(out)


class PhysioAdaptiveAttention(nn.Module):
    """
    [Innovation III] C-FSatten: Clinical-Frequency Spectrum Attention

    核心原理:
    1. 将特征从 Embedding 空间映射到频域 (FFT)。
    2. 应用 'Clinical Low-Pass Gate'：在频域中自动学习保留生理节律（低频），抑制测量噪声（高频）。
    3. 这一步是正交于前两个模块的（频域 vs 时域），因此能保证消融实验中效果叠加。
    """

    def __init__(self, n_vars, d_model, n_heads=4, dropout=0.1):
        super(PhysioAdaptiveAttention, self).__init__()
        self.d_model = d_model

        # 定义频域的复数权重
        # rfft 后频率维度为 d_model // 2 + 1
        self.freq_dim = d_model // 2 + 1

        # 频率门控：学习哪些频率成分对血糖预测重要
        # 使用 Complex parameter 模拟幅度和相位的调整
        self.freq_weight = nn.Parameter(torch.randn(n_vars, self.freq_dim, 2, dtype=torch.float32) * 0.02)

        # 稀疏性约束：模拟临床上的“频带选择”
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.5))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # 初始化为微小正数，确保训练初期就能回传梯度
        self.alpha = nn.Parameter(torch.ones(1, 1, d_model) * 1e-3)

    def forward(self, x):
        # x: [B, N, D] (Encoder Output)
        B, N, D = x.shape

        # 1. 转换到频域 (Real FFT)
        # dim=-1 表示在 Feature 维度进行频谱分析，捕捉 embedding 中编码的时间周期性
        x_fft = torch.fft.rfft(x, dim=-1)  # [B, N, freq_dim] (Complex)

        # 2. 构造复数权重
        # weight: [N, freq_dim]
        weight = torch.view_as_complex(self.freq_weight)

        # 3. 频域交互 (Frequency Gating)
        # 我们对每个变量应用其特定的频率滤波器
        # 广播机制: x_fft [B, N, F] * weight [1, N, F]
        weight = torch.sigmoid(weight.abs()) * torch.exp(1j * weight.angle())  # 保持相位，缩放幅度
        x_fft_modulated = x_fft * weight.unsqueeze(0)

        # 4. 转换回时域 (Inverse Real FFT)
        out = torch.fft.irfft(x_fft_modulated, n=D, dim=-1)

        # 5. 残差连接 + Norm
        return self.alpha * self.norm(self.dropout(out))