import numpy as np
import warnings
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import SharedBackbone
from utils.dataset import get_dataloaders


def ota_aggregate_gradients(grads, config):
    if not grads: return None
    grad_sum = np.sum(grads, axis=0)
    signal_power = np.mean(grad_sum ** 2)
    snr_linear = 10 ** (config['ota_snr_db'] / 10)
    noise_power = signal_power / snr_linear if snr_linear > 0 else 0
    noise = np.random.normal(0, np.sqrt(noise_power), grad_sum.shape)
    return grad_sum + noise


def low_rank_matrix_completion(mat):
    if np.isnan(mat).all(): return np.zeros_like(mat)
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


def compute_cmc_map_from_logits(all_logits, all_labels, max_rank=20):
    """从分类logits计算CMC和mAP（基于softmax概率排序）"""
    probs = torch.softmax(all_logits, dim=1).numpy()
    labels = all_labels.numpy()
    n_query = len(labels)

    # 按概率降序排序每个样本的类别
    sorted_indices = np.argsort(-probs, axis=1)

    # CMC
    cmc = np.zeros(max_rank)
    for rank in range(max_rank):
        if rank == 0:
            cmc[rank] = (sorted_indices[:, 0] == labels).sum()
        else:
            cmc[rank] = (sorted_indices[:, rank] == labels).sum()
    cmc = np.cumsum(cmc) / n_query

    # mAP: 对每个样本计算AP
    ap_per_query = np.zeros(n_query)
    for i in range(n_query):
        true_label = labels[i]
        rank_pos = np.where(sorted_indices[i] == true_label)[0][0]
        # AP = 1 / (rank + 1)
        ap_per_query[i] = 1.0 / (rank_pos + 1)
    mAP = ap_per_query.mean()

    return cmc, mAP


def evaluate_global_model(clients, global_backbone_state, device):
    accs, losses = [], []
    all_logits, all_labels = [], []
    criterion = torch.nn.CrossEntropyLoss()
    for client in clients:
        client.set_global_features(global_backbone_state)
        client.encoder.eval()
        client.shared_backbone.eval()
        client.classifier.eval()
        correct, total, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for img_1, img_2, labels in client.test_loader:
                if len(labels) == 0: continue
                labels = labels.to(device)
                inputs = img_1.to(device) if client.modal_type == "RGB" else img_2.to(device)
                outputs = client.classifier(client.shared_backbone(client.encoder(inputs)))
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_logits.append(outputs.cpu())
                all_labels.append(labels.cpu())
        if total > 0:
            accs.append(100. * correct / total)
            losses.append(running_loss / total)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    cmc, mAP = compute_cmc_map_from_logits(all_logits, all_labels)
    r1 = cmc[0] * 100
    return r1, cmc, mAP * 100, np.mean(losses)


def run_proposed(config):
    train_loaders, test_loaders, num_classes = get_dataloaders(config)
    global_model = SharedBackbone(config['dataset']).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = []
    for i in range(config['num_sbs'] * config['clients_per_sbs']):
        modal = "RGB" if (i % 2 == 0) else "IR"
        clients.append(ClientUE(i, config, train_loaders[i], test_loaders[i], num_classes, modal))

    sbs_cache = {i: [] for i in range(config['num_sbs'])}
    r1_history, cmc_history, map_history, loss_history = [], [], [], []
    pbar = tqdm(range(config['global_rounds']), colour='green', desc="[Proposed MM-OTA-LRMC]")

    for round_idx in pbar:
        edge_grads = []
        for sbs_id in range(config['num_sbs']):
            cluster_clients = clients[sbs_id * config['clients_per_sbs']: (sbs_id + 1) * config['clients_per_sbs']]

            raw_grads = [c.local_train(global_state_dict) for c in cluster_clients]
            ota_grad = ota_aggregate_gradients(raw_grads, config)

            if np.random.rand() < config['ota_drop_rate']:
                ota_grad = np.full_like(raw_grads[0], np.nan)

            sbs_cache[sbs_id].append(ota_grad)
            if len(sbs_cache[sbs_id]) > config['time_window_size']:
                sbs_cache[sbs_id].pop(0)

            grad_matrix = np.array(sbs_cache[sbs_id])
            completed_matrix = low_rank_matrix_completion(grad_matrix)
            edge_grads.append(completed_matrix[-1])

        global_update_1d = ota_aggregate_gradients(edge_grads, config)

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

        r1, cmc, mAP, mean_loss = evaluate_global_model(clients, global_state_dict, config['device'])
        r1_history.append(r1)
        cmc_history.append(cmc[4] * 100)  # CMC@5
        map_history.append(mAP)
        loss_history.append(mean_loss)
        pbar.set_postfix(R1=f"{r1:.2f}%", CMC5=f"{cmc[4]*100:.2f}%", mAP=f"{mAP:.2f}%")

    return {"R1": r1_history, "CMC": cmc_history, "mAP": map_history, "loss": loss_history}