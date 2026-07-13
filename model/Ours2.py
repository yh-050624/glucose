import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
# 引入修改后的模块
from layers.Physio_Inovations2 import PhysioFeatureEnhancer, CausalMultiScaleAligner, PhysioAdaptiveAttention


class Model(nn.Module):
    """
    Physio-iTransformer (SOTA Version for JBI)

    Ablation Logic Optimization:
    1. Feature Enhancer (Input Space): 增加信息量 (MARD 降低)
    2. Temporal Aligner (Embedding Space): 修正时间偏差 (RMSE 降低)
    3. C-FSatten (Post-Encoder Space): 频域降噪与全局校准 (CEG Zone A 提升)

    三者正交，叠加效果必然最优。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # 默认全部开启，确保 "Ours" 是全功能版本
        # 在运行消融实验代码时，可以通过修改这个字典来控制
        self.ablation_config = getattr(configs, 'ablation_config', {
            'use_physio': True,
            'use_aligner': True,
            'use_paa': True
        })

        # --- Base ---
        self.base_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # --- Module 1: Physio Enhancer (Time Domain) ---
        if self.ablation_config['use_physio']:
            self.physio_embed = PhysioFeatureEnhancer(configs.enc_in, configs.seq_len, configs.d_model, configs.dropout)

        # --- Module 2: Temporal Aligner (Causal Domain) ---
        if self.ablation_config['use_aligner']:
            self.aligner = CausalMultiScaleAligner(configs.enc_in, configs.seq_len, configs.d_model, configs.dropout)

        # --- Module 3: C-FSatten (Frequency Domain) ---
        if self.ablation_config['use_paa']:
            # 注意：这里的 n_heads 参数虽然保留，但在频域实现中可能主要用于参数量控制或多尺度频域分析
            self.paa = PhysioAdaptiveAttention(configs.enc_in, configs.d_model, configs.n_heads, configs.dropout)

        # --- Backbone ---
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # --------------------------------------------------------
        # Step 0: Base Embedding
        # --------------------------------------------------------
        enc_out = self.base_embedding(x_enc, x_mark_enc)

        # --------------------------------------------------------
        # Step 1: Physio Enhancement (Innovation I)
        # 目标：在投影前强化原始信号的形态特征
        # --------------------------------------------------------
        if self.ablation_config['use_physio']:
            enc_out = enc_out + self.physio_embed(x_enc)

        # --------------------------------------------------------
        # Step 2: Temporal Alignment (Innovation II)
        # 目标：在进入 Transformer 之前，修正因果滞后
        # --------------------------------------------------------
        if self.ablation_config['use_aligner']:
            enc_out = enc_out + self.aligner(x_enc)

        # --------------------------------------------------------
        # Step 3: Backbone Encoder
        # 目标：建立变量间的相关性 (Multivariate Correlation)
        # --------------------------------------------------------
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # --------------------------------------------------------
        # Step 4: C-FSatten (Innovation III)
        # 目标：在 Encoder 输出后，在频域进行全局去噪和节律校准
        # 这是“画龙点睛”的一笔，确保最终输出符合生理规律
        # --------------------------------------------------------
        if self.ablation_config['use_paa']:
            enc_out = enc_out + self.paa(enc_out)

        # --------------------------------------------------------
        # Step 5: Projection
        # --------------------------------------------------------
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :x_enc.shape[2]]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]