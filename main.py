"""Proposed算法：空中计算 + LRMC梯度补全 + 缓冲池 + 时间窗掉队"""

import numpy as np
import torch
from tqdm import tqdm
from nodes.edges import EdgeSBS
from nodes.models import create_model
from utils.dataset import get_dataloaders
from utils.common import (flatten_params, unflatten_params, local_train,
                          evaluate, clip_grad, sim_time, low_rank_matrix_completion)


def run_proposed(config):
    """Proposed：掉队者的梯度通过LRMC+缓冲池恢复，而非丢弃"""
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    device = config['device']
    total_clients = config['num_sbs'] * config['clients_per_sbs']

    model = create_model(config['dataset'], num_classes).to(device)
    p_dim = flatten_params(model).numel()
    config['grad_dim'] = p_dim

    n_samples = [len(loader.dataset) for loader in train_loaders]
    u_per = config['clients_per_sbs']
    sbs_list = [EdgeSBS(i, config) for i in range(config['num_sbs'])]

    local_model = create_model(config['dataset'], num_classes).to(device)
    grad_buffer = {}
    buffer_weight = config.get('buffer_weight', 0.3)
    recovery_weight = config.get('recovery_weight', 0.7)

    metric_history, loss_history = [], []
    pbar = tqdm(range(config['global_rounds']), colour='blue', desc="[Proposed]")

    for round_idx in pbar:
        global_lr = config.get('global_lr_init', 1.0) * (config.get('global_lr_decay', 1.0) ** round_idx)
        model.train()
        w_global = model.state_dict()
        edge_grads = []

        for sbs in sbs_list:
            start = sbs.id * u_per
            sbs_loaders = train_loaders[start:start + u_per]
            sbs_n_samples = n_samples[start:start + u_per]
            times = [sum(sim_time(n, config)) for n in sbs_n_samples]
            k = max(1, int(len(times) * (1.0 - config['straggler_rate']))) - 1
            t_win = min(config.get('t_deadline', 99999.0), sorted(times)[k])

            grad_list, is_valid = [], []
            for j, loader in enumerate(sbs_loaders):
                uav_id = start + j
                grad = local_train(local_model, loader, config, device, w_global)
                grad = clip_grad(grad)
                if times[j] > t_win:
                    hist = grad_buffer.get(uav_id)
                    grad = hist.clone() if hist is not None else torch.full_like(grad, float('nan'))
                    is_valid.append(False)
                else:
                    grad_buffer[uav_id] = grad.clone()
                    is_valid.append(True)
                grad_list.append(grad)

            has_straggler = any(not v for v in is_valid)
            if has_straggler:
                grad_matrix = torch.stack(grad_list).cpu().numpy()
                completed = low_rank_matrix_completion(grad_matrix)
                for j in range(len(is_valid)):
                    if not is_valid[j]:
                        hist = grad_buffer.get(start + j)
                        if hist is not None:
                            hist_np = hist.cpu().numpy()
                            lrmc_val = completed[j]
                            if not np.isnan(lrmc_val).all():
                                completed[j] = buffer_weight * hist_np + (1 - buffer_weight) * lrmc_val
                            else:
                                completed[j] = hist_np
                weights = np.array([1.0 if v else recovery_weight for v in is_valid])
                cluster_grad = np.average(completed, axis=0, weights=weights) * len(is_valid)
                g_tensor = torch.tensor(cluster_grad, dtype=flatten_params(model).dtype, device=device)
            else:
                g_tensor = torch.stack(grad_list).sum(dim=0)
            edge_grads.append(g_tensor)

        if edge_grads:
            g_global = torch.stack(edge_grads).sum(dim=0) / total_clients
            unflatten_params(model, flatten_params(model) - global_lr * g_global)

        acc, loss = evaluate(model, test_loader, device, config)
        metric_history.append(acc)
        loss_history.append(loss)
        pbar.set_postfix(Acc=f"{acc:.2f}%", Loss=f"{loss:.4f}")

    return {"accuracy": metric_history, "loss": loss_history}
