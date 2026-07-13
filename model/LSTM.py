import torch
import torch.nn as nn


class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        # 提取生理序列特征
        self.lstm = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout
        )

        # 序列投影：将 Context Vector 映射为完整的 Pred_Len 序列
        self.projector = nn.Linear(configs.d_model, configs.c_out * configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index=None):
        # x_enc: [B, L, N]
        out, _ = self.lstm(x_enc)

        # 取最后一个时间步作为特征总结
        last_hidden = out[:, -1, :]

        # 映射并重塑维度为 [B, Pred_Len, C_out]
        prediction = self.projector(last_hidden).view(-1, self.pred_len, 1)
        return prediction