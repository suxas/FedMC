import numpy as np
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import SharedBackbone
from utils.dataset import get_dataloaders
from main import evaluate_global_model, ota_aggregate_gradients


def run_fedavg(config):
    train_loaders, test_loaders, num_classes = get_dataloaders(config)
    global_model = SharedBackbone(config['dataset']).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = []
    for i in range(config['num_sbs'] * config['clients_per_sbs']):
        modal = "RGB" if (i % 2 == 0) else "IR"
        clients.append(ClientUE(i, config, train_loaders[i], test_loaders[i], num_classes, modal))

    is_reid = (config['dataset'] == 'SYSU-MM01')
    eval_every = config.get('eval_every', 1)
    history = {'R1': [], 'mAP': [], 'loss': []} if is_reid else {'accuracy': [], 'loss': []}
    pbar = tqdm(range(config['global_rounds']), colour='red', desc="[OTA FedAvg]")

    for round_idx in pbar:
        edge_grads = []
        for sbs_id in range(config['num_sbs']):
            cluster_clients = clients[sbs_id * config['clients_per_sbs']: (sbs_id + 1) * config['clients_per_sbs']]

            raw_grads = [c.local_train(global_state_dict) for c in cluster_clients]
            ota_grad = ota_aggregate_gradients(raw_grads, config)

            if np.random.rand() >= config['ota_drop_rate']:
                edge_grads.append(ota_grad)

        if edge_grads:
            global_update_1d = ota_aggregate_gradients(edge_grads, config)
            # NaN 保护
            if np.isnan(global_update_1d).any():
                global_update_1d = np.nan_to_num(global_update_1d, nan=0.0)
            idx = 0
            for name, param in global_state_dict.items():
                size = param.numel()
                update_tensor = torch.tensor(global_update_1d[idx:idx + size]).reshape(param.shape).to(config['device'])

                updated_val = global_state_dict[name].float() - update_tensor.float()
                if not global_state_dict[name].is_floating_point():
                    global_state_dict[name].copy_(updated_val.round().to(param.dtype))
                else:
                    global_state_dict[name].copy_(updated_val.to(param.dtype))

                idx += size

        if (round_idx + 1) % eval_every == 0 or round_idx == config['global_rounds'] - 1:
            if is_reid:
                r1, mAP, mean_loss = evaluate_global_model(clients, global_state_dict, config['device'], config)
                history['R1'].append(r1)
                history['mAP'].append(mAP)
                history['loss'].append(mean_loss)
                pbar.set_postfix(R1=f"{r1:.1f}%", mAP=f"{mAP:.1f}%", Loss=f"{mean_loss:.4f}")
            else:
                mean_acc, mean_loss = evaluate_global_model(clients, global_state_dict, config['device'], config)
                history['accuracy'].append(mean_acc)
                history['loss'].append(mean_loss)
                pbar.set_postfix(Acc=f"{mean_acc:.2f}%", Loss=f"{mean_loss:.4f}")

    return history