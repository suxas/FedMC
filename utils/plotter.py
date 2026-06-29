import matplotlib.pyplot as plt
import os


def setup_plot_style(font_size=16):
    """IEEE 论文图表风格配置"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
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


def plot_accuracy_comparison(results_dict, save_path="data/results/accuracy_comparison.pdf"):
    setup_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ['o', 's', '^', 'D']
    lines = ['-', '--', '-.', ':']

    for i, (algo_name, data) in enumerate(results_dict.items()):
        markevery = max(1, len(data['rounds']) // 10)
        ax.plot(data['rounds'], data['accuracy'],
                label=algo_name,
                marker=markers[i % len(markers)],
                linestyle=lines[i % len(lines)],
                markevery=markevery,
                alpha=0.9)

    ax.set_xlabel('Global Communication Rounds')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Performance Under System & Statistical Heterogeneity')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n[Plotter] Figure saved to: {save_path}")
    print(f"[Plotter] Figure saved to: {png_path}")
    plt.show()
    plt.close()
