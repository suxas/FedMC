import numpy as np
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.server import CloudServer
from utils.dataset import get_dataloaders


def run_fedavg(config):
    """FedAvg 基线算法

    参考DMFL实现：
    1. 掉队者基于时间阈值判断（同SBS内相对排序）
    2. 掉队者梯度直接丢弃
    3. 簇内求和，全局求和后除以总客户端数
    4. 在官方测试集上评估
    """
    # 初始化
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = []
    for i in range(config['num_sbs'] * config['clients_per_sbs']):
        clients.append(ClientUE(i, config, train_loaders[i], None, num_classes))

    lbs = CloudServer(config)
    straggler_ratio = config.get('straggler_ratio', 0.5)
    metric1_history, metric2_history, loss_history = [], [], []
    pbar = tqdm(range(config['global_rounds']), colour='red', desc="[FedAvg]")

    for round_idx in pbar:
        edge_gradients = []

        for sbs_id in range(config['num_sbs']):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            # 计算所有客户端的时间（参考DMFL）
            times = []
            for client in cluster_clients:
                _, total_time = client.simulate_communication_time(round_idx)
                times.append(total_time)
            
            # 确定时间窗口阈值（参考DMFL）
            k = max(1, int(len(cluster_clients) * (1.0 - straggler_ratio))) - 1
            t_win = sorted(times)[k]
            
            # 收集正常客户端梯度，掉队者直接丢弃
            valid_grads = []
            for client, t in zip(cluster_clients, times):
                if t <= t_win:
                    grad, _ = client.local_train(global_state_dict)
                    valid_grads.append(grad)

            if valid_grads:
                # 簇内求和（参考DMFL）
                cluster_grad = np.sum(valid_grads, axis=0)
                edge_gradients.append((cluster_grad, len(valid_grads)))

        # 全局聚合：求和后除以总客户端数（参考DMFL：除以全部客户端数，非活跃客户端数）
        if edge_gradients:
            num_total_clients = config['num_sbs'] * config['clients_per_sbs']
            global_update = sum(g for g, _ in edge_gradients) / num_total_clients
            lbs.apply_update(global_state_dict, global_update, config['device'])

        # 在官方测试集上评估（参考DMFL）
        acc, loss = lbs.evaluate(global_state_dict, test_loader, global_model, config['device'])

        metric1_history.append(acc)
        metric2_history.append(acc)
        loss_history.append(loss)

        pbar.set_postfix(Loss=f"{loss:.4f}", Acc=f"{acc:.2f}%")

    return {"loss": loss_history, "acc": metric1_history}
