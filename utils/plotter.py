import matplotlib.pyplot as plt
import os
from datetime import datetime


def setup_plot_style(font_family='Times New Roman', font_size=14):
    """配置学术论文图表风格"""
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [font_family], 'font.size': font_size,
        'axes.labelsize': font_size + 2, 'axes.titlesize': font_size + 2,
        'legend.fontsize': font_size - 2, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
        'lines.linewidth': 2.5, 'lines.markersize': 8, 'figure.autolayout': True,
        'axes.unicode_minus': False
    })


def plot_results(results_dict, save_path="data/results/comparison.png",
                 font_family='Times New Roman', font_size=14, markers=None, lines=None):
    """绘制算法对比图并保存 PNG 和 PDF（分类任务：1x2 Loss + Acc）"""
    if markers is None:
        markers = ['o', 's', '^', 'D']
    if lines is None:
        lines = ['-', '--', '-.', ':']

    setup_plot_style(font_family=font_family, font_size=font_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (algo_name, data) in enumerate(results_dict.items()):
        loss_data = data.get('loss', data.get('Loss', []))
        acc_data = data.get('acc', data.get('Acc', []))
        rounds = data.get('rounds', list(range(1, len(loss_data) + 1)))
        markevery = max(1, len(rounds) // 10)

        ax1.plot(rounds, loss_data, label=algo_name,
                 marker=markers[i % len(markers)], linestyle=lines[i % len(lines)],
                 markevery=markevery, alpha=0.85)
        ax2.plot(rounds, acc_data, label=algo_name,
                 marker=markers[i % len(markers)], linestyle=lines[i % len(lines)],
                 markevery=markevery, alpha=0.85)

    axes_config = [
        (ax1, 'Global Communication Rounds', 'Test Loss', 'Test Loss', 'upper right'),
        (ax2, 'Global Communication Rounds', 'Accuracy (%)', 'Test Accuracy', 'lower right'),
    ]

    for ax, xlabel, ylabel, title, loc in axes_config:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc=loc, framealpha=0.95)
        ax.grid(True, linestyle='--', alpha=0.6)

    # 保存
    base_dir = os.path.dirname(save_path)
    base_name = os.path.basename(save_path)
    name_without_ext, ext = os.path.splitext(base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    png_path = os.path.join(base_dir,
                            f"{name_without_ext}_{timestamp}.png") if base_dir else f"{name_without_ext}_{timestamp}.png"
    pdf_path = os.path.join(base_dir,
                            f"{name_without_ext}_{timestamp}.pdf") if base_dir else f"{name_without_ext}_{timestamp}.pdf"

    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[Plotter] 绘图已保存至:\n  - {png_path}\n  - {pdf_path}")
