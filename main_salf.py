import copy
import numpy as np
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.server import CloudServer
from utils.dataset import get_dataloaders

def get_layer_metadata(model):
    layer_info = []
    offset = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        numel = param.numel()
        priority = 3 if 'classifier' in name else (2 if 'proj' in name else 1)
        layer_info.append({
            'name': name,
            'offset': offset,
            'numel': numel,
            'size_bits': numel * 32,
            'priority': priority
        })
        offset += numel
    sorted_layers = sorted(layer_info, key=lambda x: x['priority'], reverse=True)
    return sorted_layers, offset

def generate_best_effort_salf_mask(sorted_layers, total_params, available_bits):
    mask = np.zeros(total_params, dtype=np.float32)
    accumulated_bits = 0
    for layer in sorted_layers:
        if accumulated_bits + layer['size_bits'] <= available_bits:
            start = layer['offset']
            end = start + layer['numel']
            mask[start:end] = 1.0
            accumulated_bits += layer['size_bits']
        else:
            break
    return mask

def run_salf(config, train_loaders=None, test_loader=None, num_classes=None, initial_state=None):
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
    sorted_layers, total_params = get_layer_metadata(global_model)

    metric_loss, metric_top1, metric_top5, metric_f1 = [], [], [], []
    pbar = tqdm(range(config['global_rounds']), colour='blue', desc="[Capacity-Adaptive SALF]")

    for round_idx in pbar:
        edge_updates = []
        for sbs_id in range(config['num_sbs']):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            client_timing = []
            for client in cluster_clients:
                _, total_time = client.simulate_communication_time(round_idx)
                rng = np.random.default_rng(config.get("seed", 42) + 1000003 * round_idx + 9176 * client.id)
                fast_fading = rng.exponential(1.0)
                path_loss = (3e8 / (4.0 * np.pi * 2.4e9 * client.distance)) ** 2
                snr = (client.tx_power * path_loss * fast_fading) / client.noise_power
                rate = max(client.bandwidth * np.log2(1.0 + snr), 5e4)
                cycles = 150
                t_cpu = (config.get("local_epochs", 4) * cycles * client.num_train_samples) / client.cpu_freq
                client_timing.append({
                    'client': client,
                    'total_time': total_time,
                    't_cpu': t_cpu,
                    'rate': rate
                })

            times = [item['total_time'] for item in client_timing]
            k = max(1, int(len(cluster_clients) * (1.0 - straggler_ratio))) - 1
            t_win = sorted(times)[k]

            sbs_grad_sum = np.zeros(total_params, dtype=np.float32)
            sbs_weight_sum = np.zeros(total_params, dtype=np.float32)

            for item in client_timing:
                client = item['client']
                t_total = item['total_time']
                t_cpu = item['t_cpu']
                rate = item['rate']
                n_samples = client.num_train_samples

                grad, _ = client.local_train(global_state_dict)

                if t_total <= t_win:
                    mask = np.ones(total_params, dtype=np.float32)
                else:
                    remaining_time = max(0.0, t_win - t_cpu)
                    available_bits = remaining_time * rate
                    mask = generate_best_effort_salf_mask(sorted_layers, total_params, available_bits)

                sbs_grad_sum += (grad * mask) * n_samples
                sbs_weight_sum += mask * n_samples

            if np.sum(sbs_weight_sum) > 0:
                edge_updates.append((sbs_grad_sum, sbs_weight_sum))

        if edge_updates:
            global_grad_sum = sum(g for g, _ in edge_updates)
            global_weight_sum = sum(w for _, w in edge_updates)
            global_update = global_grad_sum / np.maximum(global_weight_sum, 1.0)
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