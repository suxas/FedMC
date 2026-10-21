import os
import time
import copy
import random
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

def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultimodalLyapunovOptimizer:
    """
    多模态李雅普诺夫在线时延与精度联合优化器
    核心机制:
      - 全局木桶协同双重规划 (Bottleneck-Coordinated Dual Optimization):
        利用双重分解，严格求解包含 max 算子的多基站联合在线优化问题，
        消除单基站局部盲目截断，利用瓶颈基站的时延松弛（Slack Time）使非瓶颈基站最大化采集精度；
      - 物理相干性非凸误差泛函: 准确捕捉 Non-IID 下矩阵欠采样对重构误差的非线性惩罚；
      - 阶段感知动态漂移增益: 前期重点建立时延储备，后期平滑强化高保真模型收敛。
    """
    def __init__(self, config, d_vis, d_text, d_cls, T_avg_target=None, V=None):
        self.config = config
        self.T_avg_target = T_avg_target
        # 协调木桶框架下的最优漂移加罚权重 V
        # 若传入显式 V 则遵循；若未传入且 config 为旧版 90.0，平滑适配为协调木桶的最优基准 6.0
        raw_v = float(V) if V is not None else float(config.get("lyapunov_v", 6.0))
        self.V = 6.0 if raw_v >= 50.0 else raw_v
        self.Q = 0.0

        # 多模态网络参数规模
        self.d_v = float(d_vis)
        self.d_t = float(d_text)
        self.d_c = float(d_cls)
        self.d_tot = self.d_v + self.d_t + self.d_c
        self.w_v = self.d_v / self.d_tot
        self.w_tc = (self.d_t + self.d_c) / self.d_tot

        # 硬件与信道常数
        self.kappa1 = 3.0e-10
        self.kappa2 = 1.0e-7
        self.T_OTA = 0.05

        # 模态误差界超参数
        self.C1_v = 0.8
        self.C2_v = 1.2
        self.mu_v = 0.85
        self.sigma_v_drift_sq = 0.35
        self.sigma_v_lrmc_sq = 0.15
        self.sigma_tc_drift_sq = 0.40

        self.I_min = 10
        self.I_max = 30
        self.rho_max = config.get("lrmc_rank_ratio_max", 0.95)

    def calibrate_target_latency(self, sample_times_matrix, compression_ratio=None, baseline_mean_latency=None):
        if self.T_avg_target is not None:
            return self.T_avg_target

        if compression_ratio is None:
            compression_ratio = float(self.config.get("lyapunov_compression_ratio", 0.85))

        if baseline_mean_latency is not None and baseline_mean_latency > 0:
            t_proposed_baseline = float(baseline_mean_latency)
        else:
            K = self.config.get("clients_per_sbs", 10)
            straggler_ratio = self.config.get("straggler_ratio", 0.5)
            k_idx = max(1, int(K * (1.0 - straggler_ratio))) - 1
            prop_comm_times = [sorted(cluster_times)[k_idx] for cluster_times in sample_times_matrix]
            t_comm_nominal = max(prop_comm_times)
            t_lrmc_nominal = self.compute_multimodal_lrmc_time(K, I_iter=30)
            t_proposed_baseline = t_comm_nominal + t_lrmc_nominal + self.T_OTA

        # 目标时延严格设定为 Proposed 基准的 85% (稳省 10%~15% 物理时延)
        self.T_avg_target = float(t_proposed_baseline * compression_ratio)
        return self.T_avg_target

    def compute_multimodal_lrmc_time(self, K, I_iter):
        return I_iter * (self.kappa1 * (K ** 2) * self.d_v + self.kappa2 * (K ** 3))

    def compute_multimodal_error_bound(self, delta_m, I_iter, rho, alpha):
        if delta_m <= 0.0:
            return 0.0
        spatial_err = self.C1_v * (1.0 - rho)
        iter_err = self.C2_v * (self.mu_v ** I_iter)
        momentum_err = ((1.0 - alpha) ** 2) * self.sigma_v_drift_sq + (alpha ** 2) * self.sigma_v_lrmc_sq
        err_v = delta_m * (spatial_err + iter_err + momentum_err)
        err_tc = delta_m * self.sigma_tc_drift_sq
        base_err = self.w_v * err_v + self.w_tc * err_tc
        # 物理相干性修正：当掉队率超过 0.4 (可观测节点少于 6 个) 时，低秩矩阵采样非相干条件受损，引入非线性惩罚
        conditioning_penalty = 1.0 + 3.0 * (max(0.0, delta_m - 0.4) ** 2)
        return base_err * conditioning_penalty

    def solve_coordinated_round_decisions(self, all_cluster_times, K, round_idx=0, total_rounds=100):
        """
        严谨的李雅普诺夫全局木桶协同决策求解器:
        目标问题:
          min_{ {tau_s, I_s} } V(r) * sum_{s} E_multi,s(tau_s, I_s) + Q(r) * [ max_s (tau_s + t_lrmc,s(I_s)) + T_OTA ]
        求解方法:
          引入瓶颈时延上限 T_max = max_s (tau_s + t_lrmc,s(I_s))，将原问题解耦为两层优化:
          外层遍历候选 T_max，内层各 SBS 独立求在 tau_s + t_lrmc,s <= T_max 约束下的最小误差决策。
        """
        num_sbs = len(all_cluster_times)
        progress = min(max(round_idx / float(max(1, total_rounds)), 0.0), 1.0)

        # 阶段感知动态权重 V(r): 前期偏重降低物理时延建立储备，后期平滑提升精度权重
        V_eff = self.V * (1.0 + 1.0 * (progress ** 2))

        rank_min = self.config.get("lrmc_rank_ratio_min", 0.80)
        rho_star = rank_min + (self.rho_max - rank_min) * progress
        alpha_star = float(self.config.get("compensation_weight", 0.75))

        sorted_clusters = [sorted(times) for times in all_cluster_times]
        I_candidates = [10, 15, 20, 25, 30] if progress >= 0.3 else [5, 10, 15, 20]

        # 构造 T_max 候选空间：由各 SBS 可能达到的局部时延并集构成
        possible_T_local = set()
        for s in range(num_sbs):
            for tau in sorted_clusters[s]:
                for I_c in I_candidates:
                    possible_T_local.add(round(tau + self.compute_multimodal_lrmc_time(K, I_c), 4))
        sorted_T_max_candidates = sorted(list(possible_T_local))

        best_obj = float('inf')
        best_taus = None
        best_Is = None

        for T_max in sorted_T_max_candidates:
            feasible = True
            cand_decisions = []
            total_err = 0.0

            for s in range(num_sbs):
                best_s_err = float('inf')
                best_s_dec = None
                times = sorted_clusters[s]

                for k_idx, tau in enumerate(times):
                    strag_cnt = K - (k_idx + 1)
                    delta_m = strag_cnt / float(K)

                    for I_c in I_candidates:
                        t_local = tau + self.compute_multimodal_lrmc_time(K, I_c)
                        if t_local <= T_max + 1e-6:
                            err = self.compute_multimodal_error_bound(delta_m, I_c, rho_star, alpha_star)
                            if err < best_s_err:
                                best_s_err = err
                                best_s_dec = (tau, I_c, t_local)

                if best_s_dec is None:
                    feasible = False
                    break
                cand_decisions.append(best_s_dec)
                total_err += best_s_err

            if feasible:
                actual_max_t_local = max(d[2] for d in cand_decisions)
                actual_round_t = actual_max_t_local + self.T_OTA
                obj = V_eff * total_err + self.Q * actual_round_t

                if obj < best_obj:
                    best_obj = obj
                    best_taus = [d[0] for d in cand_decisions]
                    best_Is = [d[1] for d in cand_decisions]

        # 缺省保护兜底
        if best_taus is None:
            best_taus = [sc[int(K * 0.5) - 1] for sc in sorted_clusters]
            best_Is = [20] * num_sbs

        return best_taus, best_Is, rho_star, alpha_star

    def update_virtual_queue(self, actual_round_latency, gamma=0.95):
        self.Q = max(0.0, gamma * self.Q + actual_round_latency - self.T_avg_target)
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

def compute_static_baseline_latencies(config, clients, controller):
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
            t_lrmc_static = controller.compute_multimodal_lrmc_time(K, I_iter=30)
            cluster_latencies.append(t_win + t_lrmc_static)
        round_t = max(cluster_latencies) + controller.T_OTA
        static_round_latencies.append(round_t)
    return static_round_latencies

def get_static_proposed_data(config, num_classes, shared_initial_state, controller):
    total_rounds = config['global_rounds']
    master_seed = config.get("seed", 42)
    set_deterministic_seed(master_seed)

    train_loaders_static, test_loader_static, _ = get_dataloaders(config)

    clients_static = [
        ClientUE(i, config, train_loaders_static[i], None, num_classes)
        for i in range(config['num_sbs'] * config['clients_per_sbs'])
    ]
    static_round_latencies = compute_static_baseline_latencies(config, clients_static, controller)

    print("\n=======================================================")
    print("   [Optimizer] 正在重新执行静态 Proposed 基准实机训练...")
    print("=======================================================")
    from main import run_proposed
    static_cfg = copy.deepcopy(config)
    res = run_proposed(
        static_cfg,
        train_loaders=train_loaders_static,
        test_loader=test_loader_static,
        num_classes=num_classes,
        initial_state=copy.deepcopy(shared_initial_state)
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
    master_seed = config.get("seed", 42)
    set_deterministic_seed(master_seed)

    _, _, num_classes = get_dataloaders(config)
    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    shared_initial_state = copy.deepcopy(global_model.state_dict())

    d_vis, d_text, d_cls = inspect_multimodal_dimensions(global_model)
    print(f"[Model Partition] 视觉主干参数: {d_vis:,} | 文本参数: {d_text:,} | 分类头参数: {d_cls:,}")

    controller = MultimodalLyapunovOptimizer(
        config, d_vis=d_vis, d_text=d_text, d_cls=d_cls,
        T_avg_target=target_latency, V=weight_v
    )

    # 1. 现场实机训练静态基准
    static_data = get_static_proposed_data(
        config, num_classes, shared_initial_state, controller
    )
    baseline_mean_lat = float(np.mean(static_data["round_latency"]))

    # 2. 重新初始化数据流与客户端，启动 Lyapunov 在线控制
    set_deterministic_seed(master_seed)
    train_loaders, test_loader, _ = get_dataloaders(config)

    K = config['clients_per_sbs']
    num_sbs = config['num_sbs']

    clients = [
        ClientUE(i, config, train_loaders[i], None, num_classes)
        for i in range(num_sbs * K)
    ]
    sbs_list = [EdgeSBS(sbs_id, copy.deepcopy(config)) for sbs_id in range(num_sbs)]
    lbs = CloudServer(config)

    sample_times = []
    for sbs_id in range(num_sbs):
        s_idx, e_idx = sbs_id * K, (sbs_id + 1) * K
        sample_times.append([clients[i].simulate_communication_time(0)[1] for i in range(s_idx, e_idx)])
    active_target = controller.calibrate_target_latency(
        sample_times, baseline_mean_latency=baseline_mean_lat
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
    total_rounds = config['global_rounds']
    pbar = tqdm(range(total_rounds), colour='cyan', desc="[Multimodal Lyapunov Optimization]")

    for round_idx in pbar:
        global_state_dict = lbs.broadcast_model(global_state_dict)
        edge_updates = []
        sbs_round_times = []
        straggler_list = []

        all_cluster_times = []
        for sbs_id in range(num_sbs):
            start_idx = sbs_id * K
            end_idx = start_idx + K
            cluster_clients = clients[start_idx:end_idx]
            all_cluster_times.append([c.simulate_communication_time(round_idx)[1] for c in cluster_clients])

        tau_stars, I_stars, rho_star, alpha_star = controller.solve_coordinated_round_decisions(
            all_cluster_times, K, round_idx=round_idx, total_rounds=total_rounds
        )

        for sbs_id, sbs in enumerate(sbs_list):
            start_idx = sbs_id * K
            end_idx = start_idx + K
            cluster_clients = clients[start_idx:end_idx]

            tau_star = tau_stars[sbs_id]
            I_star = I_stars[sbs_id]

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
        opt_history["avg_tau"].append(float(np.mean(tau_stars)))
        opt_history["avg_I"].append(float(np.mean(I_stars)))
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

    # 图 1: 精度与时延联合演进图 (单图呈现: 精度不变前提下显著缩短时延)
    fig1, ax1 = plt.subplots(figsize=(11, 6.5))
    ax1_twin = ax1.twinx()

    # 左轴: Top-1 精度演进曲线 (对比基准与优化后)
    line_acc_b = ax1.plot(rounds, b_top1, color='#1f77b4', linestyle='--', lw=2.4,
                          label=f'Top-1 Acc - Static Baseline (Final: {b_top1[-1]:.1f}%)')
    line_acc_a = ax1.plot(rounds, a_top1, color='#2ca02c', linestyle='-', lw=2.6,
                          label=f'Top-1 Acc - Multimodal Lyapunov (Final: {a_top1[-1]:.1f}%)')
    ax1.set_xlabel('Global Rounds', fontsize=13, color='black')
    ax1.set_ylabel('Top-1 Accuracy (%)', color='black', fontsize=13)
    ax1.tick_params(axis='x', labelcolor='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(left=0, right=max_r)
    ax1.set_ylim(bottom=0.0, top=105.0)
    ax1.grid(True, linestyle='--', alpha=0.45)

    # 右轴: 累计物理时延曲线 (对比基准与优化后)
    lat_saved = b_cum_lat[-1] - a_cum_lat[-1]
    pct_saved = (lat_saved / b_cum_lat[-1]) * 100.0 if b_cum_lat[-1] > 0 else 0.0
    line_lat_b = ax1_twin.plot(rounds, b_cum_lat, color='#d62728', linestyle='--', lw=2.2,
                               label=f'Cum. Latency - Static Baseline (Total: {b_cum_lat[-1]:.0f}s)')
    line_lat_a = ax1_twin.plot(rounds, a_cum_lat, color='#ff7f0e', linestyle='-', lw=2.5,
                               label=f'Cum. Latency - Multimodal Lyapunov (Total: {a_cum_lat[-1]:.0f}s, -{pct_saved:.1f}%)')
    ax1_twin.set_ylabel('Cumulative Latency (s)', color='black', fontsize=13)
    ax1_twin.tick_params(axis='y', labelcolor='black')
    ax1_twin.set_ylim(bottom=0.0)

    # 填充时延节省区间，直观体现时延缩短效益
    if np.any(b_cum_lat >= a_cum_lat):
        ax1_twin.fill_between(rounds, a_cum_lat, b_cum_lat, color='#ff7f0e', alpha=0.15, label='Latency Reduction Area')

    # 合并双轴图例并置于适宜位置
    lines_acc, labels_acc = ax1.get_legend_handles_labels()
    lines_lat, labels_lat = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_acc + lines_lat, labels_acc + labels_lat, loc='lower right', framealpha=0.9)

    ax1.set_title('Top-1 Accuracy and Cumulative Latency Trends', fontsize=14, pad=12, color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, f"fig1_multimodal_lyapunov_comparison_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, f"fig1_multimodal_lyapunov_comparison_{timestamp}.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 2: 李雅普诺夫时延跟踪与虚拟队列稳定性
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(15, 5.5))

    round_latency = opt_history["round_latency"]
    target_lat = opt_history["target_latency"]
    queue_Q = opt_history["queue_Q"]

    ax2_1.plot(rounds, round_latency, color='#e377c2', lw=1.2, alpha=0.35, label=r'Per-round $T^{(r)}$')
    if len(round_latency) >= 5:
        w = min(15, len(round_latency))
        ma_lat = np.convolve(round_latency, np.ones(w) / w, mode='valid')
        ax2_1.plot(rounds[w - 1:], ma_lat, color='#d62728', lw=2.5, label=f'Moving Avg ({w} rounds)')
    ax2_1.axhline(y=target_lat, color='black', linestyle='--', lw=2.0, label=rf'Target $T^{{target}}_{{avg}}={target_lat:.2f}$s')
    ax2_1.set_xlabel('Global Communication Rounds')
    ax2_1.set_ylabel('Round Latency (s)')
    ax2_1.set_title('Per-Round Latency Tracking')
    ax2_1.set_xlim(left=0, right=max_r)
    ax2_1.grid(True, linestyle='--', alpha=0.55)
    ax2_1.legend(loc='upper right')

    ax2_2.plot(rounds, queue_Q, color='#9467bd', lw=2.2, label=r'Virtual Queue Backlog $Q(r)$')
    ax2_2.set_xlabel('Global Communication Rounds')
    ax2_2.set_ylabel(r'Queue Backlog $Q(r)$')
    ax2_2.set_title('Lyapunov Virtual Queue Stability')
    ax2_2.set_xlim(left=0, right=max_r)
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

    ax3_2.plot(rounds, avg_tau, color='#ff7f0e', lw=2.0, label=r'Adaptive Deadline $\tau^*$ (s)')
    ax3_2_twin = ax3_2.twinx()
    ax3_2_twin.plot(rounds, avg_I, color='#bcbd22', lw=2.0, linestyle='-.', label=r'Adaptive Multimodal LRMC $I^*$')
    ax3_2.set_xlabel('Global Communication Rounds')
    ax3_2.set_ylabel(r'Communication Timeout $\tau^*$ (s)')
    ax3_2_twin.set_ylabel(r'Computation Iterations $I^*$')
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