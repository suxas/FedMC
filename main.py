import numpy as np
import warnings
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import SharedBackbone
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders


# ============================================================
# 保留旧接口兼容（FedAvg/SALF 可能引用）
# ============================================================

def ota_aggregate_gradients(grads, config):
    """OTA 聚合梯度（旧接口，供 FedAvg/SALF 使用）"""
    valid_grads = [g for g in grads if g is not None]
    if not valid_grads:
        return None
    grad_sum = np.sum(valid_grads, axis=0, dtype=np.float64)
    signal_power = np.mean(grad_sum ** 2)
    snr_linear = 10 ** (config['ota_snr_db'] / 10)
    noise_power = signal_power / snr_linear if snr_linear > 0 else 0
    noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-10)), grad_sum.shape)
    return (grad_sum + noise) / len(valid_grads)


def low_rank_matrix_completion(mat):
    """LRMC 梯度补全（旧接口）"""
    if np.isnan(mat).all():
        return np.zeros_like(mat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(mat, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    inds = np.where(np.isnan(mat))

    mat_filled = np.copy(mat)
    mat_filled[inds] = np.take(col_mean, inds[1])

    U, s, Vt = np.linalg.svd(mat_filled, full_matrices=False)
    rank = max(1, int(len(s) * 0.5))
    s[rank:] = 0
    mat_reconstructed = np.dot(U, np.dot(np.diag(s), Vt))

    mat_final = np.copy(mat)
    mat_final[inds] = mat_reconstructed[inds]
    return mat_final


def evaluate_metrics(clients, global_backbone_state, device):
    """评估指标（旧接口，供 SALF 使用）"""
    cmc5_scores, r1_scores, map_scores = [], [], []

    for client in clients:
        client.set_global_features(global_backbone_state)
        if client.encoder_rgb is not None:
            client.encoder_rgb.eval()
        if client.encoder_ir is not None and client.encoder_ir is not client.encoder_rgb:
            client.encoder_ir.eval()
        if client.fusion is not None:
            client.fusion.eval()
        client.shared_backbone.eval()
        client.classifier.eval()

        correct1, correct5, total = 0, 0, 0
        client_preds, client_labels = [], []

        with torch.no_grad():
            for img_1, img_2, labels in client.test_loader:
                if len(labels) == 0:
                    continue
                labels = labels.to(device)
                img_rgb = img_1.to(device)
                img_ir = img_2.to(device)

                outputs = client._forward(img_rgb, img_ir)

                _, top5_preds = outputs.topk(5, 1, True, True)
                top5_preds = top5_preds.t()
                correct = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))

                correct1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                correct5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                total += labels.size(0)

                _, predicted = outputs.max(1)
                client_preds.extend(predicted.cpu().numpy())
                client_labels.extend(labels.cpu().numpy())

        if total > 0:
            r1_scores.append(100. * correct1 / total)
            cmc5_scores.append(100. * correct5 / total)

            preds_arr = np.array(client_preds)
            labels_arr = np.array(client_labels)
            ap_list = [1.0 if p == l else 0.0 for p, l in zip(preds_arr, labels_arr)]
            map_scores.append(np.mean(ap_list) * 100 if len(ap_list) > 0 else 0.0)

    return np.mean(cmc5_scores), np.mean(r1_scores), np.mean(map_scores)


# ============================================================
# Proposed 算法：LRMC + 缓冲池补偿 + 时间窗口掉队模拟
# ============================================================

def run_proposed(config):
    """Proposed 算法完整流程

    Step 1: LBS → SBS → UAV 广播全局模型
    Step 2: UAV 本地训练，SBS 收集梯度、LRMC 补全、缓冲池补偿、簇内聚合
    Step 3: SBS → LBS OTA 传输，全局聚合
    """
    # --- 初始化 ---
    train_loaders, test_loaders, num_classes, modal_types = get_dataloaders(config)
    global_model = SharedBackbone(config['dataset']).to(config['device'])
    global_state_dict = global_model.state_dict()

    # 创建客户端
    clients = []
    for i in range(config['num_sbs'] * config['clients_per_sbs']):
        modal = modal_types[i]
        clients.append(ClientUE(i, config, train_loaders[i], test_loaders[i], num_classes, modal))

    # 创建 SBS 和 LBS
    sbs_list = [EdgeSBS(sbs_id, config) for sbs_id in range(config['num_sbs'])]
    lbs = CloudServer(config)

    # 记录历史指标
    r1_history, map_history, loss_history = [], [], []
    eval_every = config.get('eval_every', 5)
    last_r1, last_map, last_loss = 0.0, 0.0, 0.0
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
                cluster_clients, global_state_dict
            )

            # 2b. LRMC 梯度补全
            completed_matrix = sbs.lrmc_complete(grad_matrix)

            # 2c. 缓冲池补偿 + 簇内聚合
            cluster_grad = sbs.intra_cluster_aggregate(
                completed_matrix, straggler_ids, cluster_clients
            )

            # 2d. OTA 传输到 LBS
            ota_grad = sbs.ota_to_lbs(cluster_grad)
            edge_gradients.append(ota_grad)

        # === Step 3: LBS 全局聚合 ===
        global_update = lbs.global_aggregate(edge_gradients)
        lbs.apply_update(global_state_dict, global_update, config['device'])

        # === 评估（每 eval_every 轮评估一次） ===
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
