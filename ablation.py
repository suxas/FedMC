import os
import copy
import time
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import get_base_config
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders
from utils.plotter import setup_plot_style, smooth_curve

class ProgressiveAblationEdgeSBS(EdgeSBS):
    """
    支持自顶向下逐级消融的边缘小基站 (SBS)
    """
    def __init__(self, sbs_id, config, switches=None):
        super().__init__(sbs_id, config)
        # 默认完整 Proposed 算法
        default_switches = {
            "enable_cg": True,    # 1. 真余弦相似度门控
            "enable_ea": True,    # 2. 残差模长自适应校准
            "enable_md": True,    # 3. 模态解耦隔离机制
            "enable_rd": True,    # 4. 残差漂移增量建模
            "enable_lrmc": True,  # 5. 核心低秩矩阵补全 (恒为 True)
        }
        if switches is not None:
            default_switches.update(switches)
        self.switches = default_switches

    def collect_gradients(self, clients, global_state_dict, round_idx=0, tau_override=None):
        self._current_round = round_idx
        self._last_global_state = global_state_dict
        self._current_round_gradients = {}
        gradients, straggler_ids, normal_ids = [], [], []
        self._row_client_ids = [c.id for c in clients]

        if self._vis_end_idx is None:
            v_end, t_end, tot = self._inspect_layer_indices(global_state_dict)
            self._vis_end_idx = v_end
            self._head_split_idx = t_end
            self.grad_dim = tot

        times = [c.simulate_communication_time(round_idx)[1] for c in clients]
        straggler_ratio = min(max(self.config.get("straggler_ratio", 0.5), 0.0), 1.0)
        t_win = tau_override if tau_override is not None else sorted(times)[max(1, int(len(clients) * (1.0 - straggler_ratio))) - 1]

        delta_matrix = np.full((len(clients), self.grad_dim), np.nan, dtype=np.float32)

        for row, (client, t) in enumerate(zip(clients, times)):
            if t > t_win:
                gradients.append(None)
                straggler_ids.append(client.id)
                continue

            grad, _ = client.local_train(global_state_dict)
            grad = np.asarray(grad, dtype=np.float32).reshape(-1)
            gradients.append(grad)
            normal_ids.append(client.id)
            self._current_round_gradients[client.id] = grad

            # 消融判定 4: 是否使用残差增量 (RD)
            if self.switches["enable_rd"]:
                history = self.gradient_buffer.get(client.id)
                if history and len(history) > 0:
                    delta_matrix[row] = grad - history[-1]
                else:
                    delta_matrix[row] = grad.copy()
            else:
                # 剔除 RD: 直接以绝对梯度矩阵作为低秩补全输入
                delta_matrix[row] = grad.copy()

        return delta_matrix, straggler_ids, normal_ids

    def lrmc_complete(self, delta_matrix, round_idx=None):
        delta_mat = np.asarray(delta_matrix, dtype=np.float32)
        if delta_mat.size == 0:
            return np.zeros((len(self._row_client_ids), self.grad_dim or 0), dtype=np.float32)

        if round_idx is None:
            round_idx = self._current_round

        v_end = self._vis_end_idx

        # 消融判定 3: 是否执行模态解耦隔离 (MD)
        if self.switches["enable_md"]:
            # 视觉分支提取并执行 LRMC
            v_delta = delta_mat[:, :v_end]
            completed_v_delta = self._lrmc_visual_block(v_delta, round_idx=round_idx)
            # 文本与分类头掉队行严格置零以锁定私有类别特征
            rest_delta = np.zeros((delta_mat.shape[0], delta_mat.shape[1] - v_end), dtype=np.float32)
            for r_idx in range(delta_mat.shape[0]):
                if not np.isnan(delta_mat[r_idx, v_end:]).any():
                    rest_delta[r_idx] = delta_mat[r_idx, v_end:]
            completed_delta = np.concatenate([completed_v_delta, rest_delta], axis=1)
        else:
            # 剔除 MD: 全模型所有参数（含分类头）暴力低秩补全
            completed_delta = self._lrmc_visual_block(delta_mat, round_idx=round_idx)

        # 消融判定 2: 是否执行能量校准模长对齐 (EA)
        observed_mask = ~np.isnan(delta_mat).any(axis=1)
        mean_target_norm = 1.0
        if self.switches["enable_ea"] and observed_mask.any():
            active_norms = np.linalg.norm(completed_delta[observed_mask, :v_end], axis=1)
            mean_target_norm = float(np.mean(active_norms))

        full_grad_matrix = np.empty((len(self._row_client_ids), self.grad_dim), dtype=np.float32)

        for row, cid in enumerate(self._row_client_ids):
            if cid in self._current_round_gradients:
                full_grad_matrix[row] = self._current_round_gradients[cid]
            else:
                if self.switches["enable_ea"]:
                    cur_recon_norm = np.linalg.norm(completed_delta[row, :v_end])
                    if cur_recon_norm > 1e-8 and mean_target_norm > 1e-8:
                        scale_factor = min(max(mean_target_norm / cur_recon_norm, 0.8), 1.6)
                        completed_delta[row, :v_end] *= scale_factor

                history = self.gradient_buffer.get(cid)
                if history and len(history) > 0:
                    hist_grad = history[-1]
                    if self.switches["enable_rd"]:
                        recon_grad = hist_grad + completed_delta[row]
                    else:
                        recon_grad = completed_delta[row]

                    # 消融判定 1: 是否使用真余弦门控 (CG)
                    if self.switches["enable_cg"]:
                        norm_r = np.linalg.norm(recon_grad)
                        norm_h = np.linalg.norm(hist_grad)
                        if norm_r > 0 and norm_h > 0:
                            cos = float(np.dot(recon_grad, hist_grad) / (norm_r * norm_h + 1e-12))
                            cos_c = max(-1.0, min(1.0, cos))
                            alpha = self.compensation_weight * (0.5 + 0.5 * max(0.0, cos_c))
                            full_grad_matrix[row] = alpha * recon_grad + (1.0 - alpha) * hist_grad
                        else:
                            full_grad_matrix[row] = recon_grad
                    else:
                        # 剔除 CG: 刚性权重混合
                        alpha = self.compensation_weight
                        full_grad_matrix[row] = alpha * recon_grad + (1.0 - alpha) * hist_grad
                else:
                    full_grad_matrix[row] = completed_delta[row]

        return np.nan_to_num(full_grad_matrix, nan=0.0)

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

def run_single_ablation_flow(variant_name, switches, config, train_loaders, test_loader, num_classes, shared_initial_state):
    print(f"\n==========================================================================")
    print(f"  [Progressive Ablation] 正在执行: {variant_name}")
    print(f"  当前使能开关: {json.dumps(switches)}")
    print(f"==========================================================================")

    set_deterministic_seed(config.get("seed", 42))

    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    global_model.load_state_dict(copy.deepcopy(shared_initial_state))
    global_state_dict = global_model.state_dict()

    clients = [
        ClientUE(i, config, train_loaders[i], None, num_classes)
        for i in range(config['num_sbs'] * config['clients_per_sbs'])
    ]
    sbs_list = [ProgressiveAblationEdgeSBS(sbs_id, config, switches=switches) for sbs_id in range(config['num_sbs'])]
    lbs = CloudServer(config)

    # 统计计算复杂度参数
    K = config['clients_per_sbs']
    v_params = sum(p.numel() for name, p in global_model.named_parameters() if 'visual' in name)
    tot_params = sum(p.numel() for p in global_model.parameters())
    d_compute = v_params if switches["enable_md"] else tot_params

    kappa1 = 3.0e-10
    kappa2 = 1.0e-7
    t_ota = 0.05
    i_iter = config.get("lrmc_max_iter", 30)
    t_lrmc = i_iter * (kappa1 * (K ** 2) * d_compute + kappa2 * (K ** 3))

    metric_loss, metric_top1, metric_top5, metric_f1 = [], [], [], []
    round_latencies, cum_latencies = [], []
    cumulative_time = 0.0

    total_rounds = config.get("global_rounds", 100)
    pbar = tqdm(range(total_rounds), colour='green', desc=f"[{variant_name}]")

    for round_idx in pbar:
        global_state_dict = lbs.broadcast_model(global_state_dict)
        edge_updates = []
        cluster_round_times = []

        for sbs_id, sbs in enumerate(sbs_list):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            # 记录时延
            times = [c.simulate_communication_time(round_idx)[1] for c in cluster_clients]
            k = max(1, int(len(cluster_clients) * (1.0 - config.get("straggler_ratio", 0.5)))) - 1
            t_win = sorted(times)[k]
            cluster_round_times.append(t_win + t_lrmc)

            delta_matrix, straggler_ids, _ = sbs.collect_gradients(
                cluster_clients, global_state_dict, round_idx
            )
            completed_matrix = sbs.lrmc_complete(delta_matrix, round_idx=round_idx)
            cluster_sum_grad, sbs_samples = sbs.intra_cluster_aggregate(
                completed_matrix, straggler_ids, cluster_clients
            )
            ota_grad = sbs.ota_to_lbs(cluster_sum_grad)
            edge_updates.append((ota_grad, sbs_samples))

        if edge_updates:
            total_samples = sum(s for _, s in edge_updates)
            global_update = sum(g for g, _ in edge_updates) / max(total_samples, 1)
            lbs.apply_update(global_state_dict, global_update, config['device'])

        # 挂钟时延累加
        round_t = max(cluster_round_times) + t_ota
        cumulative_time += round_t
        round_latencies.append(round_t)
        cum_latencies.append(cumulative_time)

        eval_res = lbs.evaluate(global_state_dict, test_loader, global_model, config['device'])
        metric_loss.append(eval_res['loss'])
        metric_top1.append(eval_res['top1'])
        metric_top5.append(eval_res['top5'])
        metric_f1.append(eval_res['f1'])

        pbar.set_postfix(
            Top1=f"{eval_res['top1']:.2f}%",
            F1=f"{eval_res['f1']:.2f}%",
            Lat=f"{cumulative_time:.1f}s"
        )

    return {
        "rounds": list(range(1, total_rounds + 1)),
        "top1": metric_top1,
        "top5": metric_top5,
        "f1": metric_f1,
        "loss": metric_loss,
        "round_latency": round_latencies,
        "cum_latency": cum_latencies
    }

def plot_progressive_ablation(results_dict, save_dir="result"):
    png_dir = os.path.join(save_dir, "png")
    pdf_dir = os.path.join(save_dir, "pdf")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    setup_plot_style(font_family='Times New Roman', font_size=13)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 规范色盘与阶梯线型（自深绿基准逐步褪色弱化）
    color_map = {
        "Proposed (Full)": "#1b7837",             # 墨绿 (最优基准)
        "Step 1: w/o CG": "#7fbf7b",              # 浅绿
        "Step 2: w/o (CG+EA)": "#ff7f0e",         # 橙色
        "Step 3: w/o (CG+EA+MD)": "#d62728",      # 红色 (语义污染剧变)
        "Step 4: Pure LRMC": "#756bb1"            # 紫灰 (仅保留原始LRMC)
    }
    line_map = {
        "Proposed (Full)": "-",
        "Step 1: w/o CG": "--",
        "Step 2: w/o (CG+EA)": "-.",
        "Step 3: w/o (CG+EA+MD)": ":",
        "Step 4: Pure LRMC": "--"
    }

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    sample_key = list(results_dict.keys())[0]
    rounds = np.array(results_dict[sample_key]["rounds"])
    max_round = float(np.max(rounds))

    # (a) Top-1 准确率 vs 迭代轮次
    for name, data in results_dict.items():
        c = color_map.get(name, "#333333")
        ls = line_map.get(name, "-")
        y = smooth_curve(data["top1"], factor=0.85)
        lw = 2.6 if name == "Proposed (Full)" else 2.0
        axs[0, 0].plot(rounds, y, label=name, color=c, linestyle=ls, linewidth=lw)
    axs[0, 0].set_xlabel("Global Communication Rounds")
    axs[0, 0].set_ylabel("Top-1 Accuracy (%)")
    axs[0, 0].set_title("(a) Top-1 Accuracy vs. Communication Rounds")
    axs[0, 0].set_xlim(left=0, right=max_round)
    axs[0, 0].set_ylim(bottom=0.0)
    axs[0, 0].grid(True, linestyle="--", alpha=0.55)
    axs[0, 0].legend(loc="lower right", framealpha=0.92)

    # (b) 真实物理挂钟时延下的收敛速度 (Time-to-Accuracy)
    for name, data in results_dict.items():
        c = color_map.get(name, "#333333")
        ls = line_map.get(name, "-")
        y = smooth_curve(data["top1"], factor=0.85)
        x_time = np.array(data["cum_latency"])
        lw = 2.6 if name == "Proposed (Full)" else 2.0
        axs[0, 1].plot(x_time, y, label=name, color=c, linestyle=ls, linewidth=lw)
    axs[0, 1].set_xlabel("Cumulative Wall-Clock Latency (s)")
    axs[0, 1].set_ylabel("Top-1 Accuracy (%)")
    axs[0, 1].set_title("(b) Top-1 Accuracy vs. Wall-Clock Latency (Time-to-Accuracy)")
    axs[0, 1].set_xlim(left=0)
    axs[0, 1].set_ylim(bottom=0.0)
    axs[0, 1].grid(True, linestyle="--", alpha=0.55)
    axs[0, 1].legend(loc="lower right", framealpha=0.92)

    # (c) Macro-F1 分数对比 (凸显模态解耦抗类别污染特性)
    for name, data in results_dict.items():
        c = color_map.get(name, "#333333")
        ls = line_map.get(name, "-")
        y = smooth_curve(data["f1"], factor=0.85)
        lw = 2.6 if name == "Proposed (Full)" else 2.0
        axs[1, 0].plot(rounds, y, label=name, color=c, linestyle=ls, linewidth=lw)
    axs[1, 0].set_xlabel("Global Communication Rounds")
    axs[1, 0].set_ylabel("Macro-F1 Score (%)")
    axs[1, 0].set_title("(c) Macro-F1 Score Evolution (Anti-Drift Robustness)")
    axs[1, 0].set_xlim(left=0, right=max_round)
    axs[1, 0].set_ylim(bottom=0.0)
    axs[1, 0].grid(True, linestyle="--", alpha=0.55)
    axs[1, 0].legend(loc="lower right", framealpha=0.92)

    # (d) 阶梯剥离性能定量退化分析 (柱状图)
    names = list(results_dict.keys())
    final_top1 = [float(np.mean(results_dict[k]["top1"][-5:])) for k in names]
    bar_cols = [color_map.get(k, "#333333") for k in names]

    y_pos = np.arange(len(names))
    bars = axs[1, 1].barh(y_pos, final_top1, color=bar_cols, alpha=0.85, height=0.55)
    axs[1, 1].set_yticks(y_pos)
    axs[1, 1].set_yticklabels(names)
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_xlabel("Final Top-1 Accuracy (%)")
    axs[1, 1].set_title("(d) Performance Degradation Cascade")
    axs[1, 1].grid(True, linestyle="--", alpha=0.55, axis='x')

    for bar, val in zip(bars, final_top1):
        axs[1, 1].text(val + 0.8, bar.get_y() + bar.get_height() / 2.0, f"{val:.2f}%", va='center', fontsize=11, fontweight='bold')

    axs[1, 1].set_xlim(right=max(final_top1) * 1.18)

    plt.tight_layout()
    png_path = os.path.join(png_dir, f"fig_ablation_latency_cascade_{timestamp}.png")
    pdf_path = os.path.join(pdf_dir, f"fig_ablation_latency_cascade_{timestamp}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[Ablation 成果] 逐步消融时延对比图表生成就绪:")
    print(f"  - PNG 图表: {png_path}")
    print(f"  - PDF 矢量: {pdf_path}")

def print_progressive_summary(results_dict):
    print("\n" + "=" * 98)
    print(f"{'Progressive Ablation Step':<28} | {'Final Top-1':<12} | {'Final F1':<12} | {'Wall-Clock (s)':<16} | {'Top-1 Loss'}")
    print("-" * 98)
    base_top1 = np.mean(results_dict["Proposed (Full)"]["top1"][-5:])

    for name, data in results_dict.items():
        top1 = np.mean(data["top1"][-5:])
        f1 = np.mean(data["f1"][-5:])
        total_time = data["cum_latency"][-1]
        loss_gap = base_top1 - top1
        gap_str = "Baseline" if name == "Proposed (Full)" else f"-{loss_gap:.2f}%"
        print(f"{name:<28} | {top1:>10.2f}% | {f1:>10.2f}% | {total_time:>14.1f}s | {gap_str:>10}")
    print("=" * 98 + "\n")

if __name__ == "__main__":
    base_config = get_base_config()
    base_config["global_rounds"] = 100
    set_deterministic_seed(base_config.get("seed", 42))

    train_loaders, test_loader, num_classes = get_dataloaders(base_config)
    init_model = FederatedModel(base_config['dataset'], num_classes)
    shared_initial_state = copy.deepcopy(init_model.state_dict())

    # 按照重要性自顶向下逐级消融，最后仅保留纯 LRMC 模块
    progressive_pipeline = {
        # 完整基准：全模块使能
        "Proposed (Full)": {
            "enable_cg": True, "enable_ea": True, "enable_md": True, "enable_rd": True, "enable_lrmc": True
        },
        # 步骤 1：移除真余弦门控 (CG)
        "Step 1: w/o CG": {
            "enable_cg": False, "enable_ea": True, "enable_md": True, "enable_rd": True, "enable_lrmc": True
        },
        # 步骤 2：继续移除模长对齐 (EA)
        "Step 2: w/o (CG+EA)": {
            "enable_cg": False, "enable_ea": False, "enable_md": True, "enable_rd": True, "enable_lrmc": True
        },
        # 步骤 3：继续移除模态解耦 (MD)
        "Step 3: w/o (CG+EA+MD)": {
            "enable_cg": False, "enable_ea": False, "enable_md": False, "enable_rd": True, "enable_lrmc": True
        },
        # 步骤 4：继续移除残差漂移 (RD)，最终仅保留 Pure LRMC
        "Step 4: Pure LRMC": {
            "enable_cg": False, "enable_ea": False, "enable_md": False, "enable_rd": False, "enable_lrmc": True
        }
    }

    ablation_db = {}
    for var_name, switches in progressive_pipeline.items():
        res = run_single_ablation_flow(
            variant_name=var_name,
            switches=switches,
            config=base_config,
            train_loaders=train_loaders,
            test_loader=test_loader,
            num_classes=num_classes,
            shared_initial_state=shared_initial_state
        )
        ablation_db[var_name] = res

    print_progressive_summary(ablation_db)
    plot_progressive_ablation(ablation_db, save_dir="result")