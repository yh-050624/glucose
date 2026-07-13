import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class Dataset_Ohio(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='559', target='cbg',
                 scale=True, cycle=288):
        self.seq_len, self.label_len, self.pred_len = size
        self.root_path, self.patient_id = root_path, data_path
        self.target, self.scale, self.cycle, self.flag = target, scale, cycle, flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # 加载官方指定文件
        fn = f"{self.patient_id}_training_raw.csv" if self.flag != 'test' else f"{self.patient_id}_testing_raw.csv"
        df_raw = pd.read_csv(os.path.join(self.root_path, fn))

        cols = ['basal', 'bolus', 'carbInput', 'IOB', 'COB', self.target]
        df_data = df_raw[cols].ffill().fillna(0)

        num_train = int(len(df_data) * 0.8)  # 8:2 划分验证集

        if self.flag == 'train':
            df_slice = df_data[:num_train]
        elif self.flag == 'val':
            df_slice = df_data[num_train:]
        else:
            df_slice = df_data

        if self.scale:
            train_data = df_data[:num_train].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_slice.values)
        else:
            data = df_slice.values

        self.data_x, self.data_y = data, data
        self.cycle_index = (np.arange(len(data)) % self.cycle)

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_begin = s_end - self.label_len
        return self.data_x[index:s_end], self.data_y[r_begin:s_end + self.pred_len], \
            np.zeros(1), np.zeros(1), torch.tensor(self.cycle_index[s_end - 1])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)