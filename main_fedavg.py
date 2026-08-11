import numpy as np
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
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
    train_loaders, test_loaders, num_classes = get_dataloaders(config)
    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = []
    for i in range(config['num_sbs'] * config['clients_per_sbs']):
        clients.append(ClientUE(i, config, train_loaders[i], test_loaders[i], num_classes))

    lbs = CloudServer(config)
    metric1_history, metric2_history, loss_history = [], [], []
    pbar = tqdm(range(config['global_rounds']), colour='red', desc="[FedAvg]")

    for round_idx in pbar:
        edge_gradients = []

        for sbs_id in range(config['num_sbs']):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            # 收集正常客户端梯度，掉队者直接丢弃
            valid_grads = []
            valid_weights = []
            for client in cluster_clients:
                is_straggler, _ = client.simulate_communication_time(round_idx)
                if not is_straggler:
                    grad, _ = client.local_train(global_state_dict)
                    valid_grads.append(grad)
                    valid_weights.append(client.num_train_samples)

            if valid_grads:
                # 簇内均值聚合
                cluster_grad = np.average(valid_grads, axis=0, weights=valid_weights)
                # OTA 传输
                snr_db = config.get('ota_snr_db', 0.0)
                if snr_db > 0:
                    signal_power = np.mean(cluster_grad ** 2)
                    snr_linear = 10 ** (snr_db / 10)
                    noise_power = signal_power / snr_linear
                    noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-10)), cluster_grad.shape)
                    cluster_grad = cluster_grad + noise
                edge_gradients.append((cluster_grad, sum(valid_weights)))

        # 全局聚合
        if edge_gradients:
            gradients, weights = zip(*edge_gradients)
            global_update = np.average(gradients, axis=0, weights=weights)
            lbs.apply_update(global_state_dict, global_update, config['device'])

        # 评估
        m1, m2, mean_loss = lbs.evaluate(clients, global_state_dict, config['device'])

        metric1_history.append(m1)
        metric2_history.append(m2)
        loss_history.append(mean_loss)

        pbar.set_postfix(Loss=f"{mean_loss:.4f}", Acc=f"{m1:.2f}%")

    return {"loss": loss_history, "acc": metric1_history}
