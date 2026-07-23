import numpy as np
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import SharedBackbone
from nodes.server import CloudServer
from utils.dataset import get_dataloaders


def run_fedavg(config):
    """FedAvg 基线算法

    流程：
    1. LBS 广播全局模型
    2. 各 UAV 本地训练
    3. 掉队者梯度直接丢弃
    4. 簇内均值聚合 → OTA → LBS 全局聚合
    """
    # 初始化
    train_loaders, test_loaders, num_classes, modal_types = get_dataloaders(config)
    global_model = SharedBackbone(config['dataset']).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = []
    for i in range(config['num_sbs'] * config['clients_per_sbs']):
        modal = modal_types[i]
        clients.append(ClientUE(i, config, train_loaders[i], test_loaders[i], num_classes, modal))

    lbs = CloudServer(config)
    r1_history, map_history, loss_history = [], [], []
    eval_every = config.get('eval_every', 5)
    last_r1, last_map, last_loss = 0.0, 0.0, 0.0
    pbar = tqdm(range(config['global_rounds']), colour='red', desc="[FedAvg]")

    for round_idx in pbar:
        edge_gradients = []

        for sbs_id in range(config['num_sbs']):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            # 收集正常客户端梯度，掉队者直接丢弃
            valid_grads = []
            for client in cluster_clients:
                is_straggler, _ = client.simulate_communication_time()
                if not is_straggler:
                    grad = client.local_train(global_state_dict)
                    valid_grads.append(grad)

            if valid_grads:
                # 簇内均值聚合
                cluster_grad = np.mean(valid_grads, axis=0)
                # OTA 传输
                snr_db = config.get('ota_snr_db', 30.0)
                if snr_db > 0:
                    signal_power = np.mean(cluster_grad ** 2)
                    snr_linear = 10 ** (snr_db / 10)
                    noise_power = signal_power / snr_linear
                    noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-10)), cluster_grad.shape)
                    cluster_grad = cluster_grad + noise
                edge_gradients.append(cluster_grad)

        # 全局聚合
        if edge_gradients:
            global_update = np.mean(edge_gradients, axis=0)
            lbs.apply_update(global_state_dict, global_update, config['device'])

        # 评估（每 eval_every 轮评估一次）
        if round_idx % eval_every == 0 or round_idx == config['global_rounds'] - 1:
            mean_r1, mean_map, mean_loss = lbs.evaluate(clients, global_state_dict, config['device'])
            last_r1, last_map, last_loss = mean_r1, mean_map, mean_loss
        else:
            mean_r1, mean_map, mean_loss = last_r1, last_map, last_loss

        r1_history.append(mean_r1)
        map_history.append(mean_map)
        loss_history.append(mean_loss)
        pbar.set_postfix(R1=f"{mean_r1:.2f}%", mAP=f"{mean_map:.2f}%", Loss=f"{mean_loss:.4f}")

    return {"r1": r1_history, "map": map_history, "loss": loss_history}
