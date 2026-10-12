import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def setup_plot_style(font_family='Times New Roman', font_size=14):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [font_family, 'DejaVu Serif', 'Times New Roman'],
        'font.size': font_size,
        'axes.labelsize': font_size + 2,
        'axes.titlesize': font_size + 2,
        'legend.fontsize': font_size - 2,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'lines.linewidth': 2.4,
        'lines.markersize': 6.5,
        'figure.autolayout': True,
        'axes.unicode_minus': False
    })

def smooth_curve(points, factor=0.88):
    if len(points) <= 3:
        return np.asarray(points, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    smoothed = np.empty_like(points)
    weight = 0.0
    accum = 0.0
    for i, val in enumerate(points):
        weight = weight * factor + (1.0 - factor)
        accum = accum * factor + (1.0 - factor) * val
        smoothed[i] = accum / max(weight, 1e-8)
    return smoothed

def plot_results(results_dict, save_path="result/multimodal_comparison.png",
                 font_family='Times New Roman', font_size=14, markers=None, lines=None):
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', 'p', 'X', '*']
    if lines is None:
        lines = ['-', '--', '-.', ':', '-', '--', '-.', ':']

    palette = ['#9467bd', '#ff7f0e', '#bcbd22', '#d62728', '#2ca02c']
    setup_plot_style(font_family=font_family, font_size=font_size)

    base_dir = "result"
    png_dir = os.path.join(base_dir, "png")
    pdf_dir = os.path.join(base_dir, "pdf")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    base_name = os.path.basename(save_path)
    name_without_ext, _ = os.path.splitext(base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metric_configs = [
        {
            "keys": ["top1", "acc", "Top1", "Acc"],
            "ylabel": "Top-1 Accuracy (%)",
            "title": "Top-1 Accuracy vs. Communication Rounds",
            "tag": "top1_accuracy",
            "legend_loc": "lower right",
            "ylim_bottom": 0.0
        },
        {
            "keys": ["top5", "Top5"],
            "ylabel": "Top-5 Accuracy (%)",
            "title": "Top-5 Accuracy vs. Communication Rounds",
            "tag": "top5_accuracy",
            "legend_loc": "lower right",
            "ylim_bottom": 0.0
        },
        {
            "keys": ["f1", "macro_f1", "F1"],
            "ylabel": "Macro-F1 Score (%)",
            "title": "Macro-F1 Score vs. Communication Rounds",
            "tag": "macro_f1",
            "legend_loc": "lower right",
            "ylim_bottom": 0.0
        },
        {
            "keys": ["loss", "Loss"],
            "ylabel": "Test Loss",
            "title": "Test Loss vs. Communication Rounds",
            "tag": "test_loss",
            "legend_loc": "upper right",
            "ylim_bottom": None
        }
    ]

    saved_files = []

    for m_cfg in metric_configs:
        has_metric = False
        for algo_data in results_dict.values():
            if any(k in algo_data for k in m_cfg["keys"]):
                has_metric = True
                break
        if not has_metric:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))
        max_round = 1.0

        for i, (algo_name, data) in enumerate(results_dict.items()):
            metric_series = None
            for k in m_cfg["keys"]:
                if k in data and len(data[k]) > 0:
                    metric_series = data[k]
                    break
            if metric_series is None:
                continue

            raw_y = np.asarray(metric_series, dtype=np.float64)
            rounds = np.asarray(data.get('rounds', list(range(1, len(raw_y) + 1))), dtype=np.float64)
            if len(rounds) > 0:
                max_round = max(max_round, float(np.max(rounds)))

            color = palette[i % len(palette)]
            markevery = max(1, len(rounds) // 12)
            smooth_y = smooth_curve(raw_y, factor=0.88)

            ax.plot(rounds, raw_y, color=color, alpha=0.22, linewidth=1.0, zorder=2)
            ax.fill_between(rounds, raw_y, smooth_y, color=color, alpha=0.14, zorder=1)
            ax.plot(
                rounds, smooth_y,
                label=algo_name,
                color=color,
                linestyle=lines[i % len(lines)],
                marker=markers[i % len(markers)],
                markevery=markevery,
                linewidth=2.4,
                zorder=3
            )

        ax.set_xlabel("Global Communication Rounds")
        ax.set_ylabel(m_cfg["ylabel"])
        ax.set_title(m_cfg["title"])
        ax.legend(loc=m_cfg["legend_loc"], framealpha=0.92, facecolor='white', edgecolor='none')
        ax.grid(True, linestyle='--', alpha=0.55)
        ax.set_xlim(left=0, right=max_round)
        if m_cfg.get("ylim_bottom") is not None:
            ax.set_ylim(bottom=m_cfg["ylim_bottom"])

        file_prefix = f"{name_without_ext}_{m_cfg['tag']}"
        png_path = os.path.join(png_dir, f"{file_prefix}_{timestamp}.png")
        latest_png = os.path.join(png_dir, f"{file_prefix}_latest.png")
        pdf_path = os.path.join(pdf_dir, f"{file_prefix}_{timestamp}.pdf")
        latest_pdf = os.path.join(pdf_dir, f"{file_prefix}_latest.pdf")

        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(latest_png, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(latest_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)

        saved_files.append((m_cfg["ylabel"], png_path, pdf_path))

    print(f"\n[Plotter] 图像成功保存:")
    for label, png_f, pdf_f in saved_files:
        print(f"  - [{label}]:\n      PNG: {png_f}\n      PDF: {pdf_f}")