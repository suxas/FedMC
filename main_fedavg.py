import numpy as np
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders
from nodes.models import FeatureExtractor
from main import evaluate_global_model  # 复用评估模块


def run_fedavg(config):
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    in_channels = 3 if config['dataset'] == "CIFAR10" else 1
    global_model = FeatureExtractor(in_channels).to(config['device'])
    global_state_dict = global_model.state_dict()
    config['grad_dim'] = sum(p.numel() for p in global_state_dict.values())

    server = CloudServer(config)
    sbs_list = [EdgeSBS(i, config) for i in range(config['num_sbs'])]
    clients = [ClientUE(i, config, train_loaders[i], num_classes) for i in
               range(config['num_sbs'] * config['clients_per_sbs'])]

    accuracy_history = []

    pbar = tqdm(range(config['global_rounds']), colour='green', desc="[FedAvg]")

    for round_idx in pbar:
        for c in clients: c.refresh_state()
        edge_grads = []

        for sbs in sbs_list:
            cluster_clients = clients[sbs.id * config['clients_per_sbs']: (sbs.id + 1) * config['clients_per_sbs']]
            received_grads = []

            for client in cluster_clients:
                # 严格的 OMA 与掉队者抛弃策略
                if client.is_straggler_state or client.channel_quality < config['oma_threshold']:
                    pass
                else:
                    received_grads.append(client.local_train(global_state_dict))

            if len(received_grads) > 0:
                edge_grads.append(np.mean(received_grads, axis=0))

        # MBS 全局聚合
        global_update_1d = server.global_aggregate(edge_grads)

        # 还原回模型
        idx = 0
        for name, param in global_state_dict.items():
            size = param.numel()
            update_tensor = torch.tensor(global_update_1d[idx:idx + size]).reshape(param.shape).to(config['device'])
            global_state_dict[name] -= update_tensor
            idx += size

        mean_acc, mean_loss = evaluate_global_model(clients, global_state_dict, test_loader, config['device'])
        accuracy_history.append(mean_acc)
        pbar.set_postfix(Acc=f"{mean_acc:.2f}%", Loss=f"{mean_loss:.4f}")

    return accuracy_history