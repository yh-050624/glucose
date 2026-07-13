import numpy as np

def MAE(pred, true): return np.mean(np.abs(pred - true))
def MSE(pred, true): return np.mean((pred - true) ** 2)
def RMSE(pred, true): return np.sqrt(MSE(pred, true))

def MARD(pred, true):
    """Mean Absolute Relative Difference - 血糖监测核心指标"""
    return np.mean(np.abs(pred - true) / (true + 1e-5)) * 100

def clarke_error_grid(y_true, y_pred):
    """Clarke EGA 临床安全性分析 (Zone A+B 为临床可接受区域)"""
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    n = len(y_true)
    zones = [0] * 5
    for i in range(n):
        y, py = y_true[i], y_pred[i]
        if (py <= 70 and y <= 70) or (py <= 1.2 * y and py >= 0.8 * y): zones[0] += 1
        elif (y <= 70 and py >= 180) or (y >= 240 and py <= 70): zones[4] += 1
        elif (y >= 130 and y <= 180 and py <= 70) or (y <= 180 and y >= 130 and py >= 240): zones[2] += 1
        elif (y >= 240 and py >= 70 and py <= 180) or (y <= 70 and py <= 240 and py >= 130): zones[3] += 1
        else: zones[1] += 1
    return [z / n * 100 for z in zones]

def metric(pred, true):
    mae, mse, rmse, mard = MAE(pred, true), MSE(pred, true), RMSE(pred, true), MARD(pred, true)
    zones = clarke_error_grid(true, pred)
    zone_ab = zones[0] + zones[1]
    return mae, mse, rmse, mard, zone_ab