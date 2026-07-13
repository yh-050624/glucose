import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.args = configs
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            tree_method='gpu_hist' if configs.use_gpu else 'auto',  # 适配 GPU
            gpu_id=configs.gpu if configs.use_gpu else None
        )
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_cycle=None):
        B, S, F = x_enc.shape
        x_flat = x_enc.detach().cpu().numpy().reshape(B, -1)
        try:
            preds = self.model.predict(x_flat)  # [Batch]
        except:
            # 如果模型未训练，返回 batch_x 的最后一刻值作为基准
            preds = x_flat[:, -1]

        preds = torch.from_numpy(preds).float().to(x_enc.device)
        # 广播到预测长度 (如果 pred_len > 1)
        preds = preds.unsqueeze(1).unsqueeze(2).repeat(1, self.args.pred_len, 1)
        return preds

    def fit_xgboost(self, train_loader):
        """专门用于 XGBoost 的离线训练函数"""
        X_train, y_train = [], []
        for i, (batch_x, batch_y, _, _, _) in enumerate(train_loader):
            X_train.append(batch_x.numpy().reshape(batch_x.shape[0], -1))
            # 血糖预测通常取预测段的 cbg 值
            y_train.append(batch_y[:, -self.args.pred_len:, -1].numpy().mean(axis=1))

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        self.model.fit(X_train, y_train)