"""FedAvg算法：掉队者不参与聚合，除数=参与客户端数"""

import torch
from tqdm import tqdm
from nodes.edges import EdgeSBS
from nodes.models import create_model
from utils.dataset import get_dataloaders
from utils.common import (flatten_params, unflatten_params, local_train,
                          evaluate, clip_grad, sim_time)


def run_fedavg(config):
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    device = config['device']

    model = create_model(config['dataset'], num_classes).to(device)
    p_dim = flatten_params(model).numel()
    config['grad_dim'] = p_dim

    n_samples = [len(loader.dataset) for loader in train_loaders]
    u_per = config['clients_per_sbs']
    sbs_list = [EdgeSBS(i, config) for i in range(config['num_sbs'])]

    local_model = create_model(config['dataset'], num_classes).to(device)

    metric_history, loss_history = [], []
    pbar = tqdm(range(config['global_rounds']), colour='green', desc="[FedAvg]")

    for round_idx in pbar:
        model.train()
        w_global = model.state_dict()
        edge_grads = []
        n_valid_total = 0

        for sbs in sbs_list:
            start = sbs.id * u_per
            sbs_loaders = train_loaders[start:start + u_per]
            sbs_n_samples = n_samples[start:start + u_per]
            times = [sum(sim_time(n, config)) for n in sbs_n_samples]
            k = max(1, int(len(times) * (1.0 - config['straggler_rate']))) - 1
            t_win = min(config.get('t_deadline', 99999.0), sorted(times)[k])

            valid_grads = []
            for j, loader in enumerate(sbs_loaders):
                if times[j] <= t_win:
                    grad = local_train(local_model, loader, config, device, w_global)
                    grad = clip_grad(grad)
                    valid_grads.append(grad)

            if valid_grads:
                edge_grads.append(torch.stack(valid_grads).sum(dim=0))
                n_valid_total += len(valid_grads)

        if edge_grads:
            g_global = torch.stack(edge_grads).sum(dim=0) / max(n_valid_total, 1)
            unflatten_params(model, flatten_params(model) - g_global)

        acc, loss = evaluate(model, test_loader, device, config)
        metric_history.append(acc)
        loss_history.append(loss)
        pbar.set_postfix(Acc=f"{acc:.2f}%", Loss=f"{loss:.4f}")

    return {"accuracy": metric_history, "loss": loss_history}
