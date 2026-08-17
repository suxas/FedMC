import numpy as np
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders


def run_proposed(config):
    """Proposed 算法完整流程

    Step 1: LBS → SBS → UAV 广播全局模型
    Step 2: UAV 本地训练，SBS 收集梯度、LRMC 补全、缓冲池补偿、簇内聚合
    Step 3: SBS → LBS OTA 传输，全局聚合
    Step 4: 在官方测试集上评估
    """
    # --- 初始化 ---
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    global_state_dict = global_model.state_dict()

    # 创建客户端
    clients = []
    for i in range(config['num_sbs'] * config['clients_per_sbs']):
        clients.append(ClientUE(i, config, train_loaders[i], None, num_classes))

    # 创建 SBS 和 LBS
    sbs_list = [EdgeSBS(sbs_id, config) for sbs_id in range(config['num_sbs'])]
    lbs = CloudServer(config)

    # 记录历史指标
    metric1_history, metric2_history, loss_history = [], [], []
    pbar = tqdm(range(config['global_rounds']), colour='green', desc="[Proposed LRMC-FL]")

    for round_idx in pbar:
        # === Step 1: LBS → SBS → UAV 广播全局模型 ===
        global_state_dict = lbs.broadcast_model(global_state_dict)

        # === Step 2: 各 SBS 簇内处理 ===
        edge_gradients = []
        for sbs_id, sbs in enumerate(sbs_list):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            # 2a. 收集梯度（含掉队检测）
            grad_matrix, straggler_ids, normal_ids = sbs.collect_gradients(
                cluster_clients, global_state_dict, round_idx
            )

            # 2b. LRMC 梯度补全
            completed_matrix = sbs.lrmc_complete(grad_matrix)

            # 2c. 缓冲池补偿 + 簇内聚合
            cluster_grad = sbs.intra_cluster_aggregate(
                completed_matrix, straggler_ids, cluster_clients
            )

            # 更新掉队者历史缓冲（LRMC恢复的梯度作为历史先验）
            for row, client in enumerate(cluster_clients):
                if client.id in straggler_ids:
                    history = sbs.gradient_buffer.setdefault(client.id, [])
                    history.append(completed_matrix[row].copy())
                    del history[:-sbs.buffer_size]

            # 2d. OTA 传输到 LBS
            ota_grad = sbs.ota_to_lbs(cluster_grad)
            edge_gradients.append((ota_grad, len(cluster_clients)))

        # === Step 3: LBS 全局聚合（参考DMFL：求和后除以总客户端数） ===
        if edge_gradients:
            total_clients = sum(w for _, w in edge_gradients)
            global_update = sum(g for g, _ in edge_gradients) / total_clients
            lbs.apply_update(global_state_dict, global_update, config['device'])

        # === Step 4: 在官方测试集上评估（参考DMFL） ===
        acc, loss = lbs.evaluate(global_state_dict, test_loader, global_model, config['device'])

        metric1_history.append(acc)
        metric2_history.append(acc)
        loss_history.append(loss)

        pbar.set_postfix(Loss=f"{loss:.4f}", Acc=f"{acc:.2f}%")

    return {"loss": loss_history, "acc": metric1_history}
