import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
# 引入新模块
from layers.Physio_Inovations import PhysioFeatureEnhancer, CausalMultiScaleAligner, PhysioAdaptiveAttention


class Model(nn.Module):
    """
    Physio-iTransformer (Corrected Temporal-Alignment Version)

    架构逻辑修正:
    1. Feature Enhancer: 处理 x_enc -> 加到 enc_out (Feature Space)
    2. Temporal Aligner: 处理 x_enc (在 L 维度卷积) -> 投影后加到 enc_out (Feature Space)
       * 这解决了之前在 D 维度卷积导致变量混合的严重错误。
    3. Adaptive Attention: 处理 enc_out -> 加到 enc_out (Interaction Space)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.ablation_config = {
            'use_physio': False,
            'use_aligner':  True,
            'use_paa': True
        }

        # --- Base ---
        self.base_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # --- Module 1: Physio Enhancer ---
        if self.ablation_config['use_physio']:
            self.physio_embed = PhysioFeatureEnhancer(configs.enc_in, configs.seq_len, configs.d_model, configs.dropout)

        # --- Module 2: Temporal Aligner (CMSC) ---
        if self.ablation_config['use_aligner']:
            # 输入通道为 enc_in (6)，而非 d_model
            self.aligner = CausalMultiScaleAligner(configs.enc_in, configs.seq_len, configs.d_model, configs.dropout)

        # --- Module 3: Adaptive Attention (PAA) ---
        if self.ablation_config['use_paa']:
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
        # Step 1: Physio Enhancement (Input: x_enc)
        # --------------------------------------------------------
        if self.ablation_config['use_physio']:
            # 增强形态特征
            enc_out = enc_out + self.physio_embed(x_enc)

        # --------------------------------------------------------
        # Step 2: Temporal Alignment (Input: x_enc)
        # --------------------------------------------------------
        if self.ablation_config['use_aligner']:
            # [关键修复] 输入原始 x_enc，在时间维度卷积，提取滞后特征
            # 然后投影叠加到 enc_out
            enc_out = enc_out + self.aligner(x_enc)

        # --------------------------------------------------------
        # Step 3: Backbone
        # --------------------------------------------------------
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # --------------------------------------------------------
        # Step 4: Adaptive Attention (Input: enc_out)
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