import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visual(history, true, preds, name, hist_min, pred_min):
    """
    血糖预测完整趋势可视化 (mg/dL)
    """
    plt.figure(figsize=(12, 6))
    x_hist = np.arange(0, hist_min, 5)
    x_pred = np.arange(hist_min, hist_min + pred_min, 5)

    plt.plot(x_hist, history, label=f'History ({hist_min} min)', color='#9E9E9E', alpha=0.6, linewidth=2)
    plt.plot(x_pred, true, label=f'GroundTruth (Next {pred_min} min)', color='#1976D2', linewidth=2.5)
    plt.plot(x_pred, preds, label=f'Prediction (Next {pred_min} min)', color='#D32F2F', linestyle='--', linewidth=2.5)

    # 临床警戒线
    plt.axhline(y=180, color='#FFA000', linestyle=':', alpha=0.8, label='Hyper (180 mg/dL)')
    plt.axhline(y=70, color='#7B1FA2', linestyle=':', alpha=0.8, label='Hypo (70 mg/dL)')
    plt.axvline(x=hist_min, color='#388E3C', linewidth=1.5, linestyle='-')

    plt.title('Blood Glucose Prediction Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Blood Glucose (mg/dL)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.3)

    dir_name = os.path.dirname(name)
    if dir_name and not os.path.exists(dir_name): os.makedirs(dir_name)
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_clarke_error_grid(y_true, y_pred, name):
    """
    精细化克拉克误差网格 (Colored Clarke EGA)
    严格执行非重叠切割，强化边缘分界实线，确保每个区域背景色独立。
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    fig, ax = plt.subplots(figsize=(8, 8))

    # 定义学术柔和浅色背景 (Pastel Colors)
    colors = {
        'A': '#F1F8E9',  # 极浅绿
        'B': '#F9FBE7',  # 浅黄绿
        'C': '#FFFDE7',  # 浅黄
        'D': '#FFF3E0',  # 浅橙
        'E': '#FFEBEE'  # 浅红
    }
    # 清晰的分界线配置
    boundary_style = {'edgecolor': '#000000', 'linewidth': 1.5, 'linestyle': '-', 'alpha': 1.0}

    # 1. 绘制各区域多边形（Polygon）并设置独立背景色与边界实线
    # Zone A
    zone_A_up = [[0, 0], [70 / 1.2, 70], [400 / 1.2, 400], [400, 400], [70, 70], [0, 0]]
    zone_A_low = [[0, 0], [70, 70], [400, 400], [400, 400 * 0.8], [70, 70 * 0.8], [0, 0]]
    for pts in [zone_A_up, zone_A_low]:
        ax.add_patch(patches.Polygon(pts, facecolor=colors['A'], zorder=1, **boundary_style))

    # Zone B
    zone_B_up = [[0, 70], [58.3, 70], [333.3, 400], [0, 400], [0, 70]]
    zone_B_low = [[70, 0], [400, 0], [400, 320], [70, 56], [70, 0]]
    for pts in [zone_B_up, zone_B_low]:
        ax.add_patch(patches.Polygon(pts, facecolor=colors['B'], zorder=0.5, **boundary_style))

    # Zone C
    ax.add_patch(
        patches.Polygon([[70, 180], [180, 180], [180, 400], [70, 400], [70, 180]], facecolor=colors['C'], zorder=0.4,
                        **boundary_style))
    ax.add_patch(
        patches.Polygon([[180, 70], [400, 70], [400, 0], [180, 0], [180, 70]], facecolor=colors['C'], zorder=0.4,
                        **boundary_style))

    # Zone D
    ax.add_patch(patches.Polygon([[180, 180], [180, 400], [400, 400], [180, 180]], facecolor=colors['D'], zorder=0.3,
                                 **boundary_style))
    ax.add_patch(
        patches.Polygon([[0, 180], [0, 400], [70, 400], [70, 180], [0, 180]], facecolor=colors['D'], zorder=0.3,
                        **boundary_style))

    # Zone E
    ax.add_patch(
        patches.Polygon([[0, 240], [0, 400], [70, 400], [70, 240], [0, 240]], facecolor=colors['E'], zorder=0.2,
                        **boundary_style))
    ax.add_patch(
        patches.Polygon([[240, 0], [400, 0], [400, 70], [240, 70], [240, 0]], facecolor=colors['E'], zorder=0.2,
                        **boundary_style))

    # 2. 绘制散点数据 (zorder=5 确保在背景色之上)
    ax.scatter(y_true, y_pred, marker='o', color='#212121', s=12, alpha=0.4, edgecolors='white', linewidth=0.3,
               zorder=5)

    # 3. 绘制理想参考线 (45度线)
    ax.plot([0, 400], [0, 400], color='black', linestyle='-', linewidth=1.5, zorder=6)

    # 4. 标注区域文字 (zorder=10 确保不被覆盖)
    font_args = {'fontsize': 18, 'fontweight': 'bold', 'va': 'center', 'ha': 'center', 'zorder': 10}
    ax.text(35, 35, 'A', color='#1B5E20', **font_args)
    ax.text(350, 260, 'B', color='#33691E', **font_args)
    ax.text(260, 350, 'B', color='#33691E', **font_args)
    ax.text(125, 370, 'C', color='#F57F17', **font_args)
    ax.text(160, 35, 'C', color='#F57F17', **font_args)
    ax.text(30, 370, 'E', color='#B71C1C', **font_args)
    ax.text(370, 30, 'E', color='#B71C1C', **font_args)
    ax.text(30, 220, 'D', color='#E65100', **font_args)
    ax.text(370, 150, 'D', color='#E65100', **font_args)

    ax.set_title('Clarke Error Grid Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_xlim(0, 400);
    ax.set_ylim(0, 400)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.savefig(name.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience;
        self.verbose = verbose;
        self.counter = 0
        self.best_score = None;
        self.early_stop = False;
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score;
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score;
            self.save_checkpoint(val_loss, model, path);
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if not os.path.exists(path): os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss