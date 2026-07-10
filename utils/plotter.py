import matplotlib.pyplot as plt
import os


def setup_plot_style(font_family='Times New Roman', font_size=14):
    """学术论文图表风格全局配置"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [font_family],
        'font.size': font_size,
        'axes.labelsize': font_size + 2,
        'axes.titlesize': font_size + 2,
        'legend.fontsize': font_size - 2,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'figure.autolayout': True
    })


def plot_results(
        results_dict,
        save_path="data/results/comparison.png",
        font_family='Times New Roman',
        font_size=14,
        markers=None,
        lines=None
):
    """绘制并保存算法对比结果 (1x2 子图，左 Acc，右 Loss)"""
    if markers is None:
        markers = ['o', 's', '^', 'D']
    if lines is None:
        lines = ['-', '--', '-.', ':']

    setup_plot_style(font_family=font_family, font_size=font_size)

    # 创建 1 行 2 列的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (algo_name, data) in enumerate(results_dict.items()):
        markevery = max(1, len(data['rounds']) // 10)
        marker = markers[i % len(markers)]
        linestyle = lines[i % len(lines)]

        # 1. 绘制 Accuracy
        ax1.plot(data['rounds'], data['accuracy'],
                 label=algo_name, marker=marker, linestyle=linestyle,
                 markevery=markevery, alpha=0.85)

        # 2. 绘制 Loss
        ax2.plot(data['rounds'], data['loss'],
                 label=algo_name, marker=marker, linestyle=linestyle,
                 markevery=markevery, alpha=0.85)

    # 设置左图属性 (Accuracy)
    ax1.set_xlabel('Global Communication Rounds')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy vs Rounds')
    ax1.legend(loc='lower right', framealpha=0.95)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 设置右图属性 (Loss)
    ax2.set_xlabel('Global Communication Rounds')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss vs Rounds')
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.grid(True, linestyle='--', alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存高精度图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')

    print(f"\n[Plotter] 绘图已保存至:\n  - {save_path}\n  - {pdf_path}")
    plt.close()