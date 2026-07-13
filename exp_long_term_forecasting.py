from data_loader import Dataset_Ohio
from torch.utils.data import DataLoader
from exp_basic import Exp_Basic
from train_utils.metrics import metric
from train_utils.tools import EarlyStopping, visual, plot_clarke_error_grid
import torch
import torch.nn as nn
from torch import optim
import os
import numpy as np
import time


class Exp_Long_Term_Forecast(Exp_Basic):
    def _build_model(self):
        """实例化模型并支持多 GPU，同时统计参数量"""
        model = self.model_dict[self.args.model].Model(self.args).float()

        # === 新增: 打印模型参数量 ===
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n[Model Structure] Total Parameters: {total_params / 1e6:.2f} M")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def calculate_flops(self, model):
        """
        计算模型的 FLOPs (浮点运算次数)
        需要安装 thop 库: pip install thop
        """
        try:
            from thop import profile
            from thop import clever_format

            # 创建一个 dummy input [Batch, Seq_Len, Variates]
            # 注意：根据您的 Data Loader，输入通常是 [B, L, N]
            dummy_input = torch.randn(1, self.args.seq_len, self.args.enc_in).to(self.device)
            # x_mark_enc, x_dec, x_mark_dec 通常也需要 dummy
            dummy_mark_enc = torch.randn(1, self.args.seq_len, 4).to(self.device)  # 假设4个时间特征
            dummy_dec = torch.randn(1, self.args.label_len + self.args.pred_len, self.args.enc_in).to(self.device)
            dummy_mark_dec = torch.randn(1, self.args.label_len + self.args.pred_len, 4).to(self.device)

            print("Computing FLOPs...")
            # 注意：这里只传入主要输入，具体取决于 Model forward 的参数
            # 大多数时序模型 forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
            macs, params = profile(model, inputs=(dummy_input, dummy_mark_enc, dummy_dec, dummy_mark_dec),
                                   verbose=False)
            macs, params = clever_format([macs, params], "%.3f")
            print(f"[Complexity] MACs: {macs}, Params: {params}\n")
        except ImportError:
            print("\n[Tip] Install 'thop' to calculate FLOPs: pip install thop")
        except Exception as e:
            print(f"\n[Warning] Could not calculate FLOPs: {e}")

    def _get_data(self, flag):
        """严格的数据加载隔离"""
        data_set = Dataset_Ohio(
            root_path=self.args.root_path, flag=flag, data_path=self.args.data_path,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features, target=self.args.target, cycle=self.args.cycle
        )
        data_loader = DataLoader(
            data_set, batch_size=self.args.batch_size if flag != 'test' else 1,
            shuffle=(flag == 'train'), drop_last=(flag == 'train'),
            num_workers=self.args.num_workers
        )
        return data_set, data_loader

    def _inverse_data(self, data, dataset):
        """反归一化逻辑：将模型输出转回真实血糖单位 mg/dL"""
        B, L, _ = data.shape
        dummy = np.zeros((B * L, dataset.data_x.shape[-1]))
        dummy[:, -1] = data.flatten()
        inv = dataset.scaler.inverse_transform(dummy)
        return inv[:, -1].reshape(B, L, 1)

    def vali(self, vali_data, vali_loader, criterion):
        """验证函数：计算 mg/dL 单位下的评估指标"""
        total_loss, preds_all, trues_all = [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _, batch_cycle) in enumerate(vali_loader):
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # Forward
                outputs = self.model(batch_x, None, dec_inp, None, batch_cycle.to(self.device))

                outputs = outputs[:, -self.args.pred_len:, -1:]
                batch_y = batch_y[:, -self.args.pred_len:, -1:]
                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                total_loss.append(loss)
                preds_all.append(outputs.detach().cpu().numpy())
                trues_all.append(batch_y.detach().cpu().numpy())

        p_real = self._inverse_data(np.concatenate(preds_all, axis=0), vali_data)
        t_real = self._inverse_data(np.concatenate(trues_all, axis=0), vali_data)
        mae, mse, rmse, mard, _ = metric(p_real, t_real)
        self.model.train()
        return np.average(total_loss), mae, rmse

    def train(self, setting):
        """核心训练逻辑：适配深度学习模型与机器学习模型(XGBoost)"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        # 尝试计算复杂度 (Optional)
        self.calculate_flops(self.model)

        # === 核心适配：XGboost 训练分支 (最小化改动) ===
        if self.args.model == 'XGboost':
            print(">>>>>>> Training XGBoost (Offline Mode) >>>>>>>")
            self.model.fit_xgboost(train_loader)
            torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
            return self.model

        # === 原有 PyTorch 训练流程 ===
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min', patience=5, factor=0.1)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            # === 新增: 记录 Epoch 开始时间 ===
            epoch_time = time.time()

            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y, _, _, batch_cycle) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                outputs = self.model(batch_x, None, dec_inp, None, batch_cycle.to(self.device))
                loss = criterion(outputs[:, -self.args.pred_len:, -1:], batch_y[:, -self.args.pred_len:, -1:])
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            # === 新增: 打印 Epoch 耗时 ===
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")

            v_loss, v_mae, v_rmse = self.vali(vali_data, vali_loader, criterion)
            print(f"Epoch: {epoch + 1} | Train Loss: {np.mean(train_loss):.5f} | Vali RMSE(mg/dL): {v_rmse:.2f}")
            scheduler.step(v_loss)
            early_stopping(v_loss, self.model, path)
            if early_stopping.early_stop: break

        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth'), map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        """最终评估、可视化与指标存档"""
        test_data, test_loader = self._get_data(flag='test')
        checkpoint_path = os.path.join(self.args.checkpoints, setting)
        if test:
            self.model.load_state_dict(
                torch.load(os.path.join(checkpoint_path, 'checkpoint.pth'), map_location=self.device))

        self.model.eval()

        # === 新增: 测试推理速度 (Inference Latency) ===
        start_time = time.time()

        histories, preds, trues = [], [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _, batch_cycle) in enumerate(test_loader):
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                outputs = self.model(batch_x, None, dec_inp, None, batch_cycle.to(self.device))
                histories.append(batch_x[:, :, -1:].detach().cpu().numpy())
                preds.append(outputs[:, -self.args.pred_len:, -1:].detach().cpu().numpy())
                trues.append(batch_y[:, -self.args.pred_len:, -1:].detach().cpu().numpy())

        # 计算推理总耗时
        inference_time = time.time() - start_time
        print(f"\n[Inference Speed] Total Time: {inference_time:.4f}s for {len(test_loader)} samples.")

        # 结果物理还原
        h_real = self._inverse_data(np.concatenate(histories, axis=0), test_data)
        p_real = self._inverse_data(np.concatenate(preds, axis=0), test_data)
        t_real = self._inverse_data(np.concatenate(trues, axis=0), test_data)

        # 指标计算
        mae, mse, rmse, mard, zone_ab = metric(p_real, t_real)
        hist_min, pred_min = self.args.seq_len * 5, self.args.pred_len * 5

        res_folder = os.path.join('./results/', setting)
        if not os.path.exists(res_folder): os.makedirs(res_folder)

        # 1. 趋势对比图
        visual(h_real[0, :, 0], t_real[0, :, 0], p_real[0, :, 0],
               os.path.join(res_folder, 'forecast_analysis.png'), hist_min, pred_min)

        # 2. 增强型 Clarke EGA
        plot_clarke_error_grid(t_real, p_real, os.path.join(res_folder, 'clarke_ega_analysis.png'))

        # 3. 结果保存
        result_file = os.path.join(checkpoint_path, "result_summary.txt")
        with open(result_file, 'a') as f:
            f.write(f"{setting} | Pred: {pred_min}min | MARD: {mard:.2f}% | EGA A+B: {zone_ab:.2f}% | "
                    f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | MSE: {mse:.2f}\n")

        print(f"Test Finished. Results saved in {res_folder}")
        return