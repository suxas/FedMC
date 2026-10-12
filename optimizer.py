import os
import time
import glob
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from config import get_base_config
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders
from utils.plotter import setup_plot_style, smooth_curve

class MultimodalLyapunovOptimizer:
    """
    多模态李雅普诺夫在线时延与精度联合优化器
    联合优化决策变量:
      - tau*: 无人机上行通信截止时间窗 (Communication Budget)
      - I*:   基站低秩矩阵补全迭代次数 (Computation Budget)
    多模态特性:
      - 仅在视觉参数块 d_v 上执行 LRMC，降低基站计算开销
      - 联合模态误差分解: E_multi = (d_v/d_tot)*E_v + ((d_t+d_c)/d_tot)*E_tc
    """
    def __init__(self, config, d_vis, d_text, d_cls, T_avg_target=None, V=None):
        self.config = config
        self.T_avg_target = T_avg_target
        self.V = float(V) if V is not None else float(config.get("lyapunov_v", 90.0))
        self.Q = 0.0

        # 多模态网络参数规模
        self.d_v = float(d_vis)
        self.d_t = float(d_text)
        self.d_c = float(d_cls)
        self.d_tot = self.d_v + self.d_t + self.d_c
        self.w_v = self.d_v / self.d_tot
        self.w_tc = (self.d_t + self.d_c) / self.d_tot

        # 硬件与信道系数
        self.kappa1 = 3.0e-10
        self.kappa2 = 1.0e-7
        self.T_OTA = 0.05

        # 模态误差界超参数
        self.C1_v = 0.8
        self.C2_v = 1.2
        self.mu_v = 0.85
        self.sigma_v_drift_sq = 0.35
        self.sigma_v_lrmc_sq = 0.15
        self.sigma_tc_drift_sq = 0.40  # Non-IID 下文本/分类头的私有漂移方差

        # 离散搜索边界
        self.I_min = 5
        self.I_max = 30
        self.rho_max = config.get("lrmc_rank_ratio_max", 0.95)

    def calibrate_target_latency(self, sample_times_matrix, compression_ratio=None):
        if self.T_avg_target is not None:
            return self.T_avg_target

        if compression_ratio is None:
            compression_ratio = float(self.config.get("lyapunov_compression_ratio", 0.78))

        natural_comm = max(max(cluster_times) for cluster_times in sample_times_matrix)
        t_natural_baseline = natural_comm + self.T_OTA
        self.T_avg_target = float(t_natural_baseline * compression_ratio)
        return self.T_avg_target

    def compute_multimodal_lrmc_time(self, K, I_iter):
        """仅视觉主干经历 SVD 分解，有效降低基站计算时延"""
        return I_iter * (self.kappa1 * (K ** 2) * self.d_v + self.kappa2 * (K ** 3))

    def compute_multimodal_error_bound(self, delta_m, I_iter, rho, alpha):
        if delta_m <= 0.0:
            return 0.0

        # 1. 连续视觉模态恢复误差界
        spatial_err = self.C1_v * (1.0 - rho)
        iter_err = self.C2_v * (self.mu_v ** I_iter)
        momentum_err = ((1.0 - alpha) ** 2) * self.sigma_v_drift_sq + (alpha ** 2) * self.sigma_v_lrmc_sq
        err_v = delta_m * (spatial_err + iter_err + momentum_err)

        # 2. 离散文本与私有分类头模态误差界 (锁定历史，不受 SVD 降秩伪影干扰)
        err_tc = delta_m * self.sigma_tc_drift_sq

        # 3. 参数加权的多模态联合估计误差
        return self.w_v * err_v + self.w_tc * err_tc

    def solve_sbs_decision(self, sbs_id, client_times, K):
        alpha_star = float(np.clip(
            self.sigma_v_drift_sq / (self.sigma_v_drift_sq + self.sigma_v_lrmc_sq + 1e-12),
            0.0, 1.0
        ))
        rho_star = self.rho_max
        sorted_times = sorted(client_times)

        best_obj = float('inf')
        best_tau = sorted_times[-1]
        best_I = self.I_min

        for k_idx, tau_candidate in enumerate(sorted_times):
            stragglers_count = K - (k_idx + 1)
            delta_m = stragglers_count / float(K)

            for I_candidate in range(self.I_min, self.I_max + 1, 5):
                t_lrmc = self.compute_multimodal_lrmc_time(K, I_candidate)
                t_total_sbs = tau_candidate + t_lrmc + self.T_OTA
                err_bound = self.compute_multimodal_error_bound(delta_m, I_candidate, rho_star, alpha_star)
                obj_val = self.V * err_bound + self.Q * t_total_sbs

                if obj_val < best_obj:
                    best_obj = obj_val
                    best_tau = tau_candidate
                    best_I = I_candidate

        return best_tau, best_I, rho_star, alpha_star

    def update_virtual_queue(self, actual_round_latency):
        self.Q = max(0.0, self.Q + actual_round_latency - self.T_avg_target)
        return self.Q

def inspect_multimodal_dimensions(model):
    d_v, d_t, d_c = 0, 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if 'visual' in name:
            d_v += n
        elif 'text' in name:
            d_t += n
        else:
            d_c += n
    return d_v, d_t, d_c

def get_static_proposed_data(config, train_loaders, test_loader, num_classes, initial_state, clients, controller):
    K = config['clients_per_sbs']
    num_sbs = config['num_sbs']
    total_rounds = config['global_rounds']
    straggler_ratio = config.get("straggler_ratio", 0.5)

    static_round_latencies = []
    for r in range(total_rounds):
        cluster_latencies = []
        for sbs_id in range(num_sbs):
            start_idx = sbs_id * K
            end_idx = start_idx + K
            cluster_clients = clients[start_idx:end_idx]
            times = [c.simulate_communication_time(r)[1] for c in cluster_clients]
            k = max(1, int(len(cluster_clients) * (1.0 - straggler_ratio))) - 1
            t_win = sorted(times)[k]
            t_lrmc_static = controller.compute_multimodal_lrmc_time(K, I_iter=20)
            cluster_latencies.append(t_win + t_lrmc_static)
        round_t = max(cluster_latencies) + controller.T_OTA
        static_round_latencies.append(round_t)

    log_files = glob.glob(os.path.join("result", "logs", "Proposed", "*.json"))
    if log_files:
        latest_file = max(log_files, key=os.path.getmtime)
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if len(data.get("top1", [])) >= total_rounds:
                    data["round_latency"] = static_round_latencies[:total_rounds]
                    data["cum_latency"] = list(np.cumsum(data["round_latency"]))
                    return data
        except Exception:
            pass

    print("[Optimizer] 执行静态基准 Proposed 回放...")
    from main import run_proposed
    static_cfg = copy.deepcopy(config)
    res = run_proposed(
        static_cfg,
        train_loaders=train_loaders,
        test_loader=test_loader,
        num_classes=num_classes,
        initial_state=initial_state
    )
    return {
        "rounds": list(range(1, total_rounds + 1)),
        "top1": res.get("top1", []),
        "top5": res.get("top5", []),
        "f1": res.get("f1", []),
        "loss": res.get("loss", []),
        "round_latency": static_round_latencies[:total_rounds],
        "cum_latency": list(np.cumsum(static_round_latencies[:total_rounds]))
    }

def run_multimodal_lyapunov_proposed(config, target_latency=None, weight_v=None):
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    shared_initial_state = copy.deepcopy(global_model.state_dict())

    d_vis, d_text, d_cls = inspect_multimodal_dimensions(global_model)
    print(f"[Model Partition] 视觉主干参数: {d_vis:,} | 文本参数: {d_text:,} | 分类头参数: {d_cls:,}")

    K = config['clients_per_sbs']
    num_sbs = config['num_sbs']

    clients = [
        ClientUE(i, config, train_loaders[i], None, num_classes)
        for i in range(num_sbs * K)
    ]
    sbs_list = [EdgeSBS(sbs_id, config) for sbs_id in range(num_sbs)]
    lbs = CloudServer(config)

    controller = MultimodalLyapunovOptimizer(
        config, d_vis=d_vis, d_text=d_text, d_cls=d_cls,
        T_avg_target=target_latency, V=weight_v
    )

    sample_times = []
    for sbs_id in range(num_sbs):
        s_idx, e_idx = sbs_id * K, (sbs_id + 1) * K
        sample_times.append([clients[i].simulate_communication_time(0)[1] for i in range(s_idx, e_idx)])
    active_target = controller.calibrate_target_latency(sample_times)

    static_data = get_static_proposed_data(
        config, train_loaders, test_loader, num_classes, shared_initial_state, clients, controller
    )

    global_state_dict = copy.deepcopy(shared_initial_state)
    opt_history = {
        "rounds": [],
        "top1": [],
        "top5": [],
        "f1": [],
        "loss": [],
        "round_latency": [],
        "cum_latency": [],
        "queue_Q": [],
        "avg_tau": [],
        "avg_I": [],
        "avg_straggler_ratio": [],
        "target_latency": active_target,
    }

    cumulative_time = 0.0
    pbar = tqdm(range(config['global_rounds']), colour='cyan', desc="[Multimodal Lyapunov Optimization]")

    for round_idx in pbar:
        global_state_dict = lbs.broadcast_model(global_state_dict)
        edge_updates = []
        sbs_round_times = []
        tau_list, I_list, straggler_list = [], [], []

        for sbs_id, sbs in enumerate(sbs_list):
            start_idx = sbs_id * K
            end_idx = start_idx + K
            cluster_clients = clients[start_idx:end_idx]
            times = [c.simulate_communication_time(round_idx)[1] for c in cluster_clients]

            tau_star, I_star, rho_star, alpha_star = controller.solve_sbs_decision(sbs_id, times, K)
            tau_list.append(tau_star)
            I_list.append(I_star)

            sbs.config["lrmc_max_iter"] = I_star
            sbs.compensation_weight = alpha_star

            delta_matrix, straggler_ids, _ = sbs.collect_gradients(
                cluster_clients, global_state_dict, round_idx, tau_override=tau_star
            )
            straggler_list.append(len(straggler_ids) / float(K))

            completed_matrix = sbs.lrmc_complete(delta_matrix, round_idx=round_idx)
            cluster_sum_grad, sbs_samples = sbs.intra_cluster_aggregate(
                completed_matrix, straggler_ids, cluster_clients
            )
            ota_grad = sbs.ota_to_lbs(cluster_sum_grad)
            edge_updates.append((ota_grad, sbs_samples))

            t_sbs_lrmc = controller.compute_multimodal_lrmc_time(K, I_star)
            sbs_round_times.append(tau_star + t_sbs_lrmc)

        if edge_updates:
            total_samples = sum(s for _, s in edge_updates)
            global_update = sum(g for g, _ in edge_updates) / max(total_samples, 1)
            lbs.apply_update(global_state_dict, global_update, config['device'])

        round_latency = max(sbs_round_times) + controller.T_OTA
        cumulative_time += round_latency
        current_Q = controller.update_virtual_queue(round_latency)

        eval_res = lbs.evaluate(global_state_dict, test_loader, global_model, config['device'])

        opt_history["rounds"].append(round_idx + 1)
        opt_history["top1"].append(eval_res['top1'])
        opt_history["top5"].append(eval_res['top5'])
        opt_history["f1"].append(eval_res['f1'])
        opt_history["loss"].append(eval_res['loss'])
        opt_history["round_latency"].append(round_latency)
        opt_history["cum_latency"].append(cumulative_time)
        opt_history["queue_Q"].append(current_Q)
        opt_history["avg_tau"].append(float(np.mean(tau_list)))
        opt_history["avg_I"].append(float(np.mean(I_list)))
        opt_history["avg_straggler_ratio"].append(float(np.mean(straggler_list)))

        pbar.set_postfix(
            Top1=f"{eval_res['top1']:.2f}%",
            Top5=f"{eval_res['top5']:.2f}%",
            F1=f"{eval_res['f1']:.2f}%",
            Lat=f"{round_latency:.2f}s",
            Q=f"{current_Q:.2f}"
        )

    return static_data, opt_history

def plot_three_evaluation_figures(static_data, opt_history, save_dir="result"):
    png_dir = os.path.join(save_dir, "png")
    pdf_dir = os.path.join(save_dir, "pdf")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    setup_plot_style(font_family='Times New Roman', font_size=13)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    rounds = np.array(opt_history["rounds"])
    max_r = float(np.max(rounds))

    b_top1 = smooth_curve(static_data["top1"][:len(rounds)], factor=0.88)
    a_top1 = smooth_curve(opt_history["top1"], factor=0.88)

    b_top5 = smooth_curve(static_data["top5"][:len(rounds)], factor=0.88)
    a_top5 = smooth_curve(opt_history["top5"], factor=0.88)

    b_f1 = smooth_curve(static_data["f1"][:len(rounds)], factor=0.88)
    a_f1 = smooth_curve(opt_history["f1"], factor=0.88)

    b_cum_lat = np.array(static_data["cum_latency"][:len(rounds)])
    a_cum_lat = np.array(opt_history["cum_latency"])

    # 图 1: 优化前后综合对比
    fig1, axs = plt.subplots(2, 2, figsize=(15, 11))

    axs[0, 0].plot(rounds, b_top1, color='#1f77b4', linestyle='--', lw=2.2, label='Proposed (Static Baseline: 50%)')
    axs[0, 0].plot(rounds, a_top1, color='#2ca02c', linestyle='-', lw=2.4, label='Proposed (After Opt: Multimodal Lyapunov)')
    axs[0, 0].set_xlabel('Global Communication Rounds')
    axs[0, 0].set_ylabel('Top-1 Accuracy (%)')
    axs[0, 0].set_title('Top-1 Accuracy Comparison')
    axs[0, 0].set_xlim(left=0, right=max_r)
    axs[0, 0].set_ylim(bottom=0.0)
    axs[0, 0].grid(True, linestyle='--', alpha=0.55)
    axs[0, 0].legend(loc='lower right')

    axs[0, 1].plot(rounds, b_top5, color='#1f77b4', linestyle='--', lw=2.2, label='Proposed (Static Baseline: 50%)')
    axs[0, 1].plot(rounds, a_top5, color='#2ca02c', linestyle='-', lw=2.4, label='Proposed (After Opt: Multimodal Lyapunov)')
    axs[0, 1].set_xlabel('Global Communication Rounds')
    axs[0, 1].set_ylabel('Top-5 Accuracy (%)')
    axs[0, 1].set_title('Top-5 Accuracy Comparison')
    axs[0, 1].set_xlim(left=0, right=max_r)
    axs[0, 1].set_ylim(bottom=0.0)
    axs[0, 1].grid(True, linestyle='--', alpha=0.55)
    axs[0, 1].legend(loc='lower right')

    axs[1, 0].plot(rounds, b_f1, color='#1f77b4', linestyle='--', lw=2.2, label='Proposed (Static Baseline: 50%)')
    axs[1, 0].plot(rounds, a_f1, color='#2ca02c', linestyle='-', lw=2.4, label='Proposed (After Opt: Multimodal Lyapunov)')
    axs[1, 0].set_xlabel('Global Communication Rounds')
    axs[1, 0].set_ylabel('Macro-F1 Score (%)')
    axs[1, 0].set_title('Macro-F1 Score Comparison')
    axs[1, 0].set_xlim(left=0, right=max_r)
    axs[1, 0].set_ylim(bottom=0.0)
    axs[1, 0].grid(True, linestyle='--', alpha=0.55)
    axs[1, 0].legend(loc='lower right')

    axs[1, 1].plot(b_cum_lat, b_top1, color='#1f77b4', linestyle='--', lw=2.2, label=f'Static Baseline ({b_cum_lat[-1]:.0f}s total)')
    axs[1, 1].plot(a_cum_lat, a_top1, color='#2ca02c', linestyle='-', lw=2.4, label=f'Multimodal Lyapunov ({a_cum_lat[-1]:.0f}s total)')
    axs[1, 1].set_xlabel('Cumulative Wall-Clock Latency (s)')
    axs[1, 1].set_ylabel('Top-1 Accuracy (%)')
    axs[1, 1].set_title('Top-1 Accuracy vs. Physical Latency (Time-to-Accuracy)')
    axs[1, 1].set_xlim(left=0)
    axs[1, 1].set_ylim(bottom=0.0)
    axs[1, 1].grid(True, linestyle='--', alpha=0.55)
    axs[1, 1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, f"fig1_multimodal_lyapunov_comparison_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, f"fig1_multimodal_lyapunov_comparison_{timestamp}.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 2: 李雅普诺夫时延跟踪与虚拟队列稳定性
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(15, 5.5))

    round_latency = opt_history["round_latency"]
    target_lat = opt_history["target_latency"]
    queue_Q = opt_history["queue_Q"]

    ax2_1.plot(rounds, round_latency, color='#e377c2', lw=1.2, alpha=0.35, label=r'Per-round \(T^{(r)}\)')
    if len(round_latency) >= 5:
        w = min(15, len(round_latency))
        ma_lat = np.convolve(round_latency, np.ones(w) / w, mode='valid')
        ax2_1.plot(rounds[w - 1:], ma_lat, color='#d62728', lw=2.5, label=f'Moving Avg ({w} rounds)')
    ax2_1.axhline(y=target_lat, color='black', linestyle='--', lw=2.0, label=rf'Target \(T^{{target}}_{{avg}}={target_lat:.2f}\)s')
    ax2_1.set_xlabel('Global Communication Rounds')
    ax2_1.set_ylabel('Round Latency (s)')
    ax2_1.set_title('Per-Round Latency Tracking')
    ax2_1.set_xlim(left=0, right=max_r)
    ax2_1.grid(True, linestyle='--', alpha=0.55)
    ax2_1.legend(loc='upper right')

    ax2_2.plot(rounds, queue_Q, color='#9467bd', lw=2.2, label=r'Virtual Queue Backlog \(Q(r)\)')
    ax2_2.set_xlabel('Global Communication Rounds')
    ax2_2.set_ylabel(r'Queue Backlog \(Q(r)\)')
    ax2_2.set_title('Lyapunov Virtual Queue Stability')
    ax2_2.set_xlim(left=0, right=max_r)
    ax2_2.set_ylim(bottom=0.0)
    ax2_2.grid(True, linestyle='--', alpha=0.55)
    ax2_2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, f"fig2_multimodal_queue_stability_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, f"fig2_multimodal_queue_stability_{timestamp}.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 3: 协同决策驱动掉队弹性变化图
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(15, 5.5))

    dyn_stragglers = np.array(opt_history["avg_straggler_ratio"]) * 100.0
    avg_tau = opt_history["avg_tau"]
    avg_I = opt_history["avg_I"]

    ax3_1.plot(rounds, dyn_stragglers, color='#17becf', lw=1.2, alpha=0.35, label='Per-round Straggler Ratio')
    if len(dyn_stragglers) >= 5:
        w = min(15, len(dyn_stragglers))
        ma_stragglers = np.convolve(dyn_stragglers, np.ones(w) / w, mode='valid')
        ax3_1.plot(rounds[w - 1:], ma_stragglers, color='#008b8b', lw=2.5, label=f'Moving Avg ({w} rounds)')
    ax3_1.axhline(y=50.0, color='#d62728', linestyle='--', lw=2.0, label='Static 50% Straggler Baseline')
    ax3_1.set_xlabel('Global Communication Rounds')
    ax3_1.set_ylabel('Straggler Ratio (%)')
    ax3_1.set_title('Dynamic Straggler Ratio Variation')
    ax3_1.set_xlim(left=0, right=max_r)
    ax3_1.set_ylim(0.0, 100.0)
    ax3_1.grid(True, linestyle='--', alpha=0.55)
    ax3_1.legend(loc='upper right')

    ax3_2.plot(rounds, avg_tau, color='#ff7f0e', lw=2.0, label=r'Adaptive Deadline \(\tau^*\) (s)')
    ax3_2_twin = ax3_2.twinx()
    ax3_2_twin.plot(rounds, avg_I, color='#bcbd22', lw=2.0, linestyle='-.', label=r'Adaptive Multimodal LRMC \(I^*\)')
    ax3_2.set_xlabel('Global Communication Rounds')
    ax3_2.set_ylabel(r'Communication Timeout \(\tau^*\) (s)')
    ax3_2_twin.set_ylabel(r'Computation Iterations \(I^*\)')
    ax3_2.set_title('Coordinated Decisions Driving Elastic Trade-off')
    ax3_2.set_xlim(left=0, right=max_r)
    ax3_2.grid(True, linestyle='--', alpha=0.55)

    lines_1, labels_1 = ax3_2.get_legend_handles_labels()
    lines_2, labels_2 = ax3_2_twin.get_legend_handles_labels()
    ax3_2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, f"fig3_multimodal_straggler_dynamics_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, f"fig3_multimodal_straggler_dynamics_{timestamp}.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    base_cfg = get_base_config()
    static_results, opt_results = run_multimodal_lyapunov_proposed(base_cfg)
    plot_three_evaluation_figures(static_results, opt_results, save_dir="result")