import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F

from config import get_base_config
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders
from utils.plotter import setup_plot_style


# ==============================================================================
# 1. 李雅普诺夫在线动态控制器 (Lyapunov Dynamic Controller)
# ==============================================================================

class LyapunovOptimizer:
    """基于李雅普诺夫动态控制的 LRMC-FL 精度-时延联合自适应优化器"""

    def __init__(self, config, T_avg_target=None, V=25.0):
        self.config = config
        self.T_avg_target = T_avg_target  # 长期平均时延硬约束 (秒)，为 None 时自动标定
        self.V = float(V)  # 漂移加罚权重
        self.Q = 0.0  # 虚拟时延队列 Q(r)

        # 硬件与时延常数标定 (对齐 problem_optimizer.md)
        self.kappa1 = 1.5e-8  # 矩阵乘法硬件算力常数
        self.kappa2 = 2.0e-6  # 特征值分解硬件算力常数
        self.T_OTA = 0.05  # 空中计算/基站聚合传输时延 (秒)

        # 误差模型先验常数
        self.C1 = 0.8
        self.C2 = 1.2
        self.mu = 0.85  # LRMC 几何收敛因子
        self.sigma_drift_sq = 0.35  # 历史梯度时序漂移方差
        self.sigma_lrmc_sq = 0.15  # 空间重构截断方差

        # 决策边界约束
        self.I_min = 5
        self.I_max = 30
        self.rho_max = config.get("lrmc_rank_ratio", 0.8)

    def calibrate_target_latency(self, sample_times_matrix, compression_ratio=0.90):
        """若未显式指定目标时延，根据系统自然时延中位数自动标定紧约束目标"""
        if self.T_avg_target is not None:
            return self.T_avg_target

        natural_latencies = []
        for cluster_times in sample_times_matrix:
            natural_latencies.append(max(cluster_times))
        base_latency = max(natural_latencies) + self.T_OTA + 0.02
        # 设定在自然最大时延的 88%~92%，施加适度时延控制压力
        self.T_avg_target = float(base_latency * compression_ratio)
        return self.T_avg_target

    def compute_lrmc_time(self, K, d_bb, I_iter):
        """计算 SBS 端执行 I_iter 次 LRMC 补全的计算耗时"""
        return I_iter * (self.kappa1 * (K ** 2) * d_bb + self.kappa2 * (K ** 3))

    def compute_error_bound(self, delta_m, I_iter, rho, alpha):
        """计算 SBS 簇内梯度重构均方误差上界 epsilon_m^2"""
        if delta_m <= 0.0:
            return 0.0
        spatial_err = self.C1 * (1.0 - rho)
        iter_err = self.C2 * (self.mu ** I_iter)
        momentum_err = ((1.0 - alpha) ** 2) * self.sigma_drift_sq + (alpha ** 2) * self.sigma_lrmc_sq
        return delta_m * (spatial_err + iter_err + momentum_err)

    def solve_sbs_decision(self, sbs_id, client_times, K, d_bb):
        """单轮李雅普诺夫决策求解: 确定最优截断门限 tau* 与补全迭代 I*"""
        alpha_star = float(np.clip(
            self.sigma_drift_sq / (self.sigma_drift_sq + self.sigma_lrmc_sq + 1e-12),
            0.0, 1.0
        ))
        rho_star = self.rho_max

        sorted_times = sorted(client_times)
        best_obj = float('inf')
        best_tau = sorted_times[-1]
        best_I = self.I_min

        # 遍历离散到达时延点进行一维搜索
        for k_idx, tau_candidate in enumerate(sorted_times):
            stragglers_count = K - (k_idx + 1)
            delta_m = stragglers_count / float(K)

            for I_candidate in range(self.I_min, self.I_max + 1, 5):
                t_lrmc = self.compute_lrmc_time(K, d_bb, I_candidate)
                t_total_sbs = tau_candidate + t_lrmc + self.T_OTA
                err_bound = self.compute_error_bound(delta_m, I_candidate, rho_star, alpha_star)

                # 李雅普诺夫漂移加罚目标: min V * Error + Q(r) * Latency
                obj_val = self.V * err_bound + self.Q * t_total_sbs

                if obj_val < best_obj:
                    best_obj = obj_val
                    best_tau = tau_candidate
                    best_I = I_candidate

        return best_tau, best_I, rho_star, alpha_star

    def update_virtual_queue(self, actual_round_latency):
        """更新虚拟队列 Q(r+1) = max(0, Q(r) + T(r) - T_target)"""
        self.Q = max(0.0, self.Q + actual_round_latency - self.T_avg_target)
        return self.Q


# ==============================================================================
# 2. 动态自适应训练执行引擎
# ==============================================================================

def run_lyapunov_proposed(config, target_latency=None, weight_v=25.0):
    """运行基于李雅普诺夫控制的自适应 Proposed 算法"""
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    global_state_dict = global_model.state_dict()

    K = config['clients_per_sbs']
    num_sbs = config['num_sbs']
    clients = [
        ClientUE(i, config, train_loaders[i], None, num_classes)
        for i in range(num_sbs * K)
    ]
    sbs_list = [EdgeSBS(sbs_id, config) for sbs_id in range(num_sbs)]
    lbs = CloudServer(config)

    controller = LyapunovOptimizer(config, T_avg_target=target_latency, V=weight_v)

    d_total = sum(p.numel() for p in global_model.parameters())
    d_bb = sum(
        p.numel() for name, p in global_model.named_parameters() if 'fc' not in name and 'classifier' not in name)

    # 预探测自然时延基准
    sample_times = []
    for sbs_id in range(num_sbs):
        s_idx, e_idx = sbs_id * K, (sbs_id + 1) * K
        sample_times.append([clients[i].simulate_communication_time(0)[1] for i in range(s_idx, e_idx)])
    active_target = controller.calibrate_target_latency(sample_times, compression_ratio=0.90)

    print(f"[Optimizer] 激活时延硬约束目标 T_avg_target: {active_target:.3f}s (漂移权重 V={weight_v})")

    history = {
        "rounds": [],
        "acc": [],
        "loss": [],
        "round_latency": [],
        "cum_latency": [],
        "queue_Q": [],
        "avg_tau": [],
        "avg_straggler_ratio": [],
        "target_latency": active_target,
    }

    cumulative_time = 0.0
    pbar = tqdm(range(config['global_rounds']), colour='cyan', desc=f"[Lyapunov LRMC-FL]")

    for round_idx in pbar:
        global_state_dict = lbs.broadcast_model(global_state_dict)
        edge_updates = []
        sbs_round_times = []
        tau_list, straggler_list = [], []

        for sbs_id, sbs in enumerate(sbs_list):
            start_idx = sbs_id * K
            end_idx = start_idx + K
            cluster_clients = clients[start_idx:end_idx]

            # 1. 状态感知
            times = [c.simulate_communication_time(round_idx)[1] for c in cluster_clients]

            # 2. 李雅普诺夫在线求解
            tau_star, I_star, rho_star, alpha_star = controller.solve_sbs_decision(sbs_id, times, K, d_bb)
            tau_list.append(tau_star)

            sbs.config["lrmc_max_iter"] = I_star
            sbs.config["lrmc_rank_ratio"] = rho_star
            sbs.compensation_weight = alpha_star

            # 3. 本地训练与动态门限截断
            gradients, straggler_ids = [], []
            sbs._row_client_ids = [c.id for c in cluster_clients]

            for client, t in zip(cluster_clients, times):
                if t > tau_star:
                    gradients.append(None)
                    straggler_ids.append(client.id)
                else:
                    grad, split_idx = client.local_train(global_state_dict)
                    gradients.append(grad)
                    if sbs.grad_dim is None:
                        sbs.grad_dim = grad.size
                    if sbs._head_split_idx is None:
                        sbs._head_split_idx = split_idx
                    h = sbs.gradient_buffer.setdefault(client.id, [])
                    h.append(grad.copy())
                    del h[:-sbs.buffer_size]

            straggler_list.append(len(straggler_ids) / float(K))

            # 4. 补全与簇内聚合
            grad_matrix = np.full((K, sbs.grad_dim or d_total), np.nan, dtype=np.float32)
            for row, g in enumerate(gradients):
                if g is not None:
                    grad_matrix[row] = g

            completed_matrix = sbs.lrmc_complete(grad_matrix)
            cluster_sum_grad, sbs_samples = sbs.intra_cluster_aggregate(
                completed_matrix, straggler_ids, cluster_clients
            )
            ota_grad = sbs.ota_to_lbs(cluster_sum_grad)
            edge_updates.append((ota_grad, sbs_samples))

            t_sbs_lrmc = controller.compute_lrmc_time(K, d_bb, I_star)
            sbs_round_times.append(tau_star + t_sbs_lrmc)

        # 5. 全局聚合
        if edge_updates:
            total_samples = sum(s for _, s in edge_updates)
            global_update = sum(g for g, _ in edge_updates) / max(total_samples, 1)
            lbs.apply_update(global_state_dict, global_update, config['device'])

        # 6. 时延结算与虚拟队列演化
        round_latency = max(sbs_round_times) + controller.T_OTA
        cumulative_time += round_latency
        current_Q = controller.update_virtual_queue(round_latency)

        # 7. 模型评估
        acc, loss = lbs.evaluate(global_state_dict, test_loader, global_model, config['device'])

        history["rounds"].append(round_idx + 1)
        history["acc"].append(acc)
        history["loss"].append(loss)
        history["round_latency"].append(round_latency)
        history["cum_latency"].append(cumulative_time)
        history["queue_Q"].append(current_Q)
        history["avg_tau"].append(float(np.mean(tau_list)))
        history["avg_straggler_ratio"].append(float(np.mean(straggler_list)))

        pbar.set_postfix(
            Acc=f"{acc:.2f}%",
            Loss=f"{loss:.4f}",
            Lat=f"{round_latency:.2f}s",
            Q=f"{current_Q:.2f}"
        )

    return history


# ==============================================================================
# 3. 轨迹可视化绘图函数 (Trajectory Plotter)
# ==============================================================================

def plot_optimizer_trajectories(history, save_dir="data/results"):
    """绘制并保存模型精确度与系统时延控制的两张学术轨迹图表"""
    os.makedirs(save_dir, exist_ok=True)
    setup_plot_style(font_family='Times New Roman', font_size=14)

    rounds = history["rounds"]
    acc = history["acc"]
    loss = history["loss"]
    round_latency = history["round_latency"]
    cum_latency = history["cum_latency"]
    queue_Q = history["queue_Q"]
    target_latency = history["target_latency"]

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # --------------------------------------------------------------------------
    # 图 1: 模型精确度与损失收敛轨迹图 (Accuracy & Loss Trajectories)
    # --------------------------------------------------------------------------
    fig1, (ax1_1, ax1_2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1a. Accuracy vs Rounds
    ax1_1.plot(rounds, acc, color='#1f77b4', lw=2.2, label='Test Accuracy')
    ax1_1.set_xlabel('Global Communication Rounds')
    ax1_1.set_ylabel('Test Accuracy (%)')
    ax1_1.set_title('Test Accuracy vs. Rounds')
    ax1_1.grid(True, linestyle='--', alpha=0.5)
    ax1_1.legend(loc='lower right')

    # 1b. Accuracy vs Execution Time (Wall-Clock Latency)
    ax1_2.plot(cum_latency, acc, color='#2ca02c', lw=2.2, label='Accuracy vs. Time')
    ax1_2.set_xlabel('Cumulative Wall-Clock Latency (s)')
    ax1_2.set_ylabel('Test Accuracy (%)')
    ax1_2.set_title('Test Accuracy vs. Cumulative Latency')
    ax1_2.grid(True, linestyle='--', alpha=0.5)
    ax1_2.legend(loc='lower right')

    plt.tight_layout()
    acc_png_path = os.path.join(save_dir, f"trajectory_accuracy_{timestamp}.png")
    acc_pdf_path = os.path.join(save_dir, f"trajectory_accuracy_{timestamp}.pdf")
    plt.savefig(acc_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(acc_pdf_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------
    # 图 2: 动态时延跟踪与李雅普诺夫队列稳定性轨迹图 (Latency & Queue Dynamics)
    # --------------------------------------------------------------------------
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 2a. Per-Round Latency Tracking
    ax2_1.plot(rounds, round_latency, color='#e377c2', lw=1.5, alpha=0.6, label='Instant Latency $T^{(r)}$')
    if len(round_latency) >= 5:
        window = min(10, len(round_latency))
        ma_latency = np.convolve(round_latency, np.ones(window) / window, mode='valid')
        ax2_1.plot(rounds[window - 1:], ma_latency, color='#d62728', lw=2.5, label=f'Moving Avg ({window} rounds)')
    ax2_1.axhline(y=target_latency, color='black', linestyle='--', lw=2.0,
                  label=f'Target $T^{{target}}_{{avg}}={target_latency:.2f}s$')
    ax2_1.set_xlabel('Global Communication Rounds')
    ax2_1.set_ylabel('Latency per Round (s)')
    ax2_1.set_title('Per-Round Latency Tracking')
    ax2_1.grid(True, linestyle='--', alpha=0.5)
    ax2_1.legend(loc='upper right')

    # 2b. Virtual Queue Q(r) Backlog Stability
    ax2_2.plot(rounds, queue_Q, color='#9467bd', lw=2.2, label='Virtual Queue $Q(r)$')
    ax2_2.set_xlabel('Global Communication Rounds')
    ax2_2.set_ylabel('Queue Backlog $Q(r)$')
    ax2_2.set_title('Lyapunov Virtual Queue Stability')
    ax2_2.grid(True, linestyle='--', alpha=0.5)
    ax2_2.legend(loc='upper right')

    plt.tight_layout()
    lat_png_path = os.path.join(save_dir, f"trajectory_latency_{timestamp}.png")
    lat_pdf_path = os.path.join(save_dir, f"trajectory_latency_{timestamp}.pdf")
    plt.savefig(lat_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(lat_pdf_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[Optimizer] 轨迹图表保存成功:")
    print(f"  1. 模型精确度变化轨迹图 -> {acc_png_path}")
    print(f"  2. 系统时延动态变化轨迹图 -> {lat_png_path}")


# ==============================================================================
# 4. 主入口执行脚本
# ==============================================================================

if __name__ == "__main__":
    config = get_base_config()

    # 动态权衡参数：V 越大越偏好精度，V 越小越严格压制时延
    WEIGHT_V = 25.0

    # 显式设定目标时延（若设为 None 则按自然时延的 90% 自动标定）
    TARGET_LATENCY = 1.38

    print("\n" + "=" * 70)
    print("  UAV-HFL 自适应李雅普诺夫优化 (Lyapunov Dynamic Control Optimizer)")
    print(f"  数据集: {config['dataset']} | 全局轮数: {config['global_rounds']}")
    print(f"  时延约束目标: {TARGET_LATENCY if TARGET_LATENCY else '自动自适应标定'}")
    print(f"  李雅普诺夫权衡权重 V: {WEIGHT_V}")
    print("=" * 70 + "\n")

    history = run_lyapunov_proposed(config, target_latency=TARGET_LATENCY, weight_v=WEIGHT_V)
    plot_optimizer_trajectories(history, save_dir="data/results")

    avg_lat = np.mean(history["round_latency"])
    print(
        f"\n[Summary] 最终准确率: {history['acc'][-1]:.2f}% | 实际平均时延: {avg_lat:.3f}s (目标: {history['target_latency']:.3f}s)")