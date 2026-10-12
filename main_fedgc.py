import copy
import numpy as np
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.server import CloudServer
from utils.dataset import get_dataloaders

def run_fedgc(config, train_loaders=None, test_loader=None, num_classes=None, initial_state=None):
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
    momentum_lambda = config.get('fedgc_momentum', 0.8)

    total_params = sum(p.numel() for p in global_model.parameters())
    client_grad_buffer = {c.id: None for c in clients}

    metric_loss, metric_top1, metric_top5, metric_f1 = [], [], [], []
    pbar = tqdm(range(config['global_rounds']), colour='magenta', desc="[Multimodal FedGC]")

    for round_idx in pbar:
        edge_updates = []
        for sbs_id in range(config['num_sbs']):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            times = [c.simulate_communication_time(round_idx)[1] for c in cluster_clients]
            k = max(1, int(len(cluster_clients) * (1.0 - straggler_ratio))) - 1
            t_win = sorted(times)[k]

            active_grads = []
            active_weights = []
            straggler_clients = []

            for client, t in zip(cluster_clients, times):
                if t <= t_win:
                    grad, _ = client.local_train(global_state_dict)
                    active_grads.append(grad)
                    active_weights.append(client.num_train_samples)
                    client_grad_buffer[client.id] = grad.copy()
                else:
                    straggler_clients.append(client)

            if len(active_grads) > 0:
                mean_active_grad = np.average(active_grads, axis=0, weights=active_weights)
            else:
                mean_active_grad = np.zeros(total_params, dtype=np.float32)

            sbs_grad_sum = np.zeros(total_params, dtype=np.float32)
            sbs_samples = 0

            for g, w in zip(active_grads, active_weights):
                sbs_grad_sum += g * w
                sbs_samples += w

            # 一阶动量外推补偿掉队者梯度
            for sc in straggler_clients:
                hist_g = client_grad_buffer[sc.id]
                if hist_g is not None:
                    imputed_grad = hist_g + momentum_lambda * (mean_active_grad - hist_g)
                else:
                    imputed_grad = mean_active_grad.copy()

                sbs_grad_sum += imputed_grad * sc.num_train_samples
                sbs_samples += sc.num_train_samples

                # 后台真实异步训练以更新其历史梯度
                async_grad, _ = sc.local_train(global_state_dict)
                client_grad_buffer[sc.id] = async_grad.copy()

            if sbs_samples > 0:
                edge_updates.append((sbs_grad_sum, sbs_samples))

        if edge_updates:
            total_s = sum(s for _, s in edge_updates)
            global_update = sum(g for g, _ in edge_updates) / max(total_s, 1)
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