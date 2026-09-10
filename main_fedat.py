import copy
import numpy as np
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.server import CloudServer
from utils.dataset import get_dataloaders

def run_fedat(config, train_loaders=None, test_loader=None, num_classes=None, initial_state=None):
    """
    FedAT (Asynchronous Tiered Federated Learning)
    规范来源: IEEE/ACM SC 2020 & IEEE/ACM Transactions on Networking 2021
    机制:
      - 梯队内同步聚合，梯队间异步更新
      - Tier 1 (快梯队): 保持每轮全局时钟同步上传
      - Tier 2 (慢梯队/掉队者): 异步跨轮计算，完成后施加多项式陈旧度惩罚衰减 alpha(tau) = (tau + 1)^(-gamma)
    """
    if train_loaders is None or test_loader is None:
        train_loaders, test_loader, num_classes = get_dataloaders(config)

    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    if initial_state is not None:
        global_model.load_state_dict(copy.deepcopy(initial_state))
    global_state_dict = global_model.state_dict()

    clients = [
        ClientUE(i, config, train_loaders[i], None, num_classes)
        for i in range(config['num_sbs'] * config['clients_per_sbs'])
    ]
    lbs = CloudServer(config)
    straggler_ratio = config.get('straggler_ratio', 0.5)

    # FedAT 异步控制变量
    tier_period = max(1, config.get("fedat_tier_period", 2))
    gamma = config.get("fedat_gamma", 0.5)
    tier2_start_state = copy.deepcopy(global_state_dict)
    tier2_start_round = 0

    metric_loss, metric_top1, metric_top5, metric_f1 = [], [], [], []
    pbar = tqdm(range(config['global_rounds']), colour='cyan', desc="[Multimodal FedAT]")

    for round_idx in pbar:
        # 判断本轮慢梯队是否刚好跨轮计算完毕
        is_tier2_ready = ((round_idx - tier2_start_round) >= tier_period) or (round_idx == config['global_rounds'] - 1)

        fast_sum_grad = 0.0
        fast_samples = 0
        slow_sum_grad = 0.0
        slow_samples = 0

        for sbs_id in range(config['num_sbs']):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            times = [c.simulate_communication_time(round_idx)[1] for c in cluster_clients]
            k = max(1, int(len(cluster_clients) * (1.0 - straggler_ratio))) - 1
            t_win = sorted(times)[k]

            for client, t in zip(cluster_clients, times):
                n_samples = client.num_train_samples
                if t <= t_win:
                    # Tier 1 (快梯队): 同步基于最新 global_state_dict 训练
                    grad, _ = client.local_train(global_state_dict)
                    fast_sum_grad += grad * n_samples
                    fast_samples += n_samples
                else:
                    # Tier 2 (慢梯队): 基于初始版本 tier2_start_state 异步长周期训练
                    if is_tier2_ready:
                        grad, _ = client.local_train(tier2_start_state)
                        slow_sum_grad += grad * n_samples
                        slow_samples += n_samples

        if is_tier2_ready and slow_samples > 0:
            # 触发陈旧度感知加权聚合
            tau = round_idx - tier2_start_round
            alpha_damping = float((tau + 1.0) ** (-gamma))

            combined_grad = fast_sum_grad + alpha_damping * slow_sum_grad
            combined_weight = fast_samples + alpha_damping * slow_samples
            global_update = combined_grad / max(combined_weight, 1.0)
            lbs.apply_update(global_state_dict, global_update, config['device'])

            # 慢梯队提交后，立即拉取最新模型开启下一个异步迭代周期
            tier2_start_state = copy.deepcopy(global_state_dict)
            tier2_start_round = round_idx + 1
        elif fast_samples > 0:
            # 慢梯队尚在计算中，中央仅同步快梯队
            global_update = fast_sum_grad / max(fast_samples, 1)
            lbs.apply_update(global_state_dict, global_update, config['device'])

        eval_res = lbs.evaluate(global_state_dict, test_loader, global_model, config['device'])
        metric_loss.append(eval_res['loss'])
        metric_top1.append(eval_res['top1'])
        metric_top5.append(eval_res['top5'])
        metric_f1.append(eval_res['f1'])

        pbar.set_postfix(
            Loss=f"{eval_res['loss']:.4f}",
            Top1=f"{eval_res['top1']:.2f}%",
            Top5=f"{eval_res['top5']:.2f}%",
            F1=f"{eval_res['f1']:.2f}%"
        )

    return {
        "loss": metric_loss,
        "top1": metric_top1,
        "top5": metric_top5,
        "f1": metric_f1,
        "acc": metric_top1
    }