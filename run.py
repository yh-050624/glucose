import argparse
import random
import numpy as np
import torch
from exp_long_term_forecasting import Exp_Long_Term_Forecast

if __name__ == '__main__':
    fix_seed = 2026
    random.seed(fix_seed);
    torch.manual_seed(fix_seed);
    np.random.seed(fix_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(fix_seed)

    parser = argparse.ArgumentParser(description='Clinical BG Prediction Benchmark')

    # === 基础配置 ===
    parser.add_argument('--model', type=str, default='iTransformer')#XGboost，LSTM，Transformer，iTransformer,Informer，Reformer，Flowformer，Flashformer，EMAformer
    parser.add_argument('--root_path', type=str, default=r'F:\glucose\data\CSDI')
    parser.add_argument('--data_path', type=str, default='559', help='Patient ID')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0')

    parser.add_argument('--seq_len', type=int, default=36, help='180 min history')
    parser.add_argument('--label_len', type=int, default=18)
    parser.add_argument('--pred_len', type=int, default=12, help='60 min prediction')

    parser.add_argument('--enc_in', type=int, default=6)
    parser.add_argument('--dec_in', type=int, default=6)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--target', type=str, default='cbg')
    parser.add_argument('--cycle', type=int, default=288)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--class_strategy', type=str, default='projection')
    parser.add_argument('--output_attention', action='store_true', default=False)
    parser.add_argument('--use_norm', type=bool, default=True)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--freq', type=str, default='5T')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--output_proj_dropout', type=float, default=0.1)
    parser.add_argument('--channel_independence', type=int, default=0, help='0: dependence, 1: independence')
    parser.add_argument('--distil', action='store_false', help='encoder distillation', default=True)
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')


    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--inverse', type=bool, default=True)
    parser.add_argument('--checkpoints', type=str, default='checkpoints/')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--features', type=str, default='M')

    args = parser.parse_args()


    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    exp = Exp_Long_Term_Forecast(args)
    setting = f"{args.model}_{args.data_path}_sl{args.seq_len}_pl{args.pred_len}"

    print(f'>>>>>>> Start Training: {setting} >>>>>>>')
    exp.train(setting)
    print(f'>>>>>>> Testing: {setting} <<<<<<<')
    exp.test(setting)
    torch.cuda.empty_cache()