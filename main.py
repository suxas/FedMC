import numpy as np
import warnings
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import SharedBackbone
from utils.dataset import get_dataloaders


def ota_aggregate_gradients(grads, config):
    if not grads: return None
    grad_sum = np.sum(grads, axis=0, dtype=np.float64)
    signal_power = np.mean(grad_sum ** 2)
    snr_linear = 10 ** (config['ota_snr_db'] / 10)
    noise_power = signal_power / snr_linear if snr_linear > 0 else 0
    noise = np.random.normal(0, np.sqrt(noise_power), grad_sum.shape)
    return (grad_sum + noise) / len(grads)


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


def evaluate_metrics(clients, global_backbone_state, device):
    cmc5_scores, r1_scores, map_scores = [], [], []

    for client in clients:
        client.set_global_features(global_backbone_state)
        client.encoder.eval()
        client.shared_backbone.eval()
        client.classifier.eval()

        correct1, correct5, total = 0, 0, 0
        client_preds, client_labels = [], []

        with torch.no_grad():
            for img_1, img_2, labels in client.test_loader:
                if len(labels) == 0: continue
                labels = labels.to(device)
                inputs = img_1.to(device) if client.modal_type == "RGB" else img_2.to(device)

                outputs = client.classifier(client.shared_backbone(client.encoder(inputs)))

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


def run_proposed(config):
    train_loaders, test_loaders, num_classes = get_dataloaders(config)
    global_model = SharedBackbone(config['dataset']).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = []
    for i in range(config['num_sbs'] * config['clients_per_sbs']):
        modal = "RGB" if (i % 2 == 0) else "IR"
        clients.append(ClientUE(i, config, train_loaders[i], test_loaders[i], num_classes, modal))

    sbs_cache = {i: [] for i in range(config['num_sbs'])}
    cmc5_history, r1_history, map_history = [], [], []
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
            # 【FedBN 服务器端】：不处理 BN 统计量
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                idx += size
                continue

            update_tensor = torch.tensor(global_update_1d[idx:idx + size]).reshape(param.shape).to(config['device'])
            updated_val = global_state_dict[name].float() - update_tensor.float()

            if not global_state_dict[name].is_floating_point():
                global_state_dict[name].copy_(updated_val.round().to(param.dtype))
            else:
                global_state_dict[name].copy_(updated_val.to(param.dtype))

            idx += size

        mean_cmc5, mean_r1, mean_map = evaluate_metrics(clients, global_state_dict, config['device'])
        cmc5_history.append(mean_cmc5)
        r1_history.append(mean_r1)
        map_history.append(mean_map)
        pbar.set_postfix(CMC5=f"{mean_cmc5:.2f}%", R1=f"{mean_r1:.2f}%", mAP=f"{mean_map:.2f}%")

    return {"cmc5": cmc5_history, "r1": r1_history, "map": map_history}