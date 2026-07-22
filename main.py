import numpy as np
import warnings
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import SharedBackbone
from utils.dataset import get_dataloaders


def ota_aggregate_gradients(grads, config):
    if not grads: return None
    # 过滤掉 NaN 梯度
    valid_grads = [g for g in grads if not np.isnan(g).any()]
    if not valid_grads:
        return np.zeros_like(grads[0])
    grad_sum = np.sum(valid_grads, axis=0)
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


def evaluate_global_model(clients, global_backbone_state, device, config=None):
    """评估：MNIST/CIFAR10 返回 (acc, loss)，SYSU-MM01 返回 (R1, mAP, loss)"""
    ds_name = config['dataset'] if config else 'MNIST'
    is_reid = (ds_name == 'SYSU-MM01')

    criterion = torch.nn.CrossEntropyLoss()

    if is_reid:
        # Re-ID 评估：收集所有客户端的特征，计算 R1/mAP
        all_feats, all_labels, all_mods = [], [], []
        total_loss, total_samples = 0.0, 0

        for client in clients:
            client.set_global_features(global_backbone_state)
            client.encoder.eval()
            client.shared_backbone.eval()
            client.classifier.eval()
            with torch.no_grad():
                for img_1, img_2, labels in client.test_loader:
                    if len(labels) == 0: continue
                    labels_dev = labels.to(device)
                    # 根据客户端模态类型提取特征
                    if client.modal_type == "RGB":
                        feat = client.shared_backbone(client.encoder(img_1.to(device)))
                    else:
                        feat = client.shared_backbone(client.encoder(img_2.to(device)))
                    all_feats.append(feat.cpu())
                    all_labels.extend(labels.tolist())
                    all_mods.extend([client.modal_type] * len(labels))
                    # loss（确保在同一设备上，处理 NaN）
                    outputs = client.classifier(feat.to(device))
                    batch_loss = criterion(outputs, labels_dev).item()
                    if not np.isnan(batch_loss):
                        total_loss += batch_loss * len(labels)
                        total_samples += len(labels)
                    else:
                        total_samples += len(labels)

        if not all_feats or total_samples == 0:
            return 0.0, 0.0, 0.0

        feats = torch.nn.functional.normalize(torch.cat(all_feats), dim=1).numpy()
        labels_arr = np.array(all_labels)
        mods_arr = np.array(all_mods)

        # IR→RGB 检索
        q_mask = mods_arr == 'IR'
        g_mask = mods_arr == 'RGB'
        q_feats, g_feats = feats[q_mask], feats[g_mask]
        q_labels, g_labels = labels_arr[q_mask], labels_arr[g_mask]

        n_gallery = len(g_labels)
        if n_gallery == 0 or len(q_labels) == 0:
            return 0.0, 0.0, total_loss / max(total_samples, 1)

        cmc = np.zeros(n_gallery)
        aps = []
        for i in range(len(q_labels)):
            dist = np.linalg.norm(q_feats[i:i+1] - g_feats, axis=1)
            sorted_idx = np.argsort(dist)
            matches = (g_labels[sorted_idx] == q_labels[i]).astype(int)
            if matches.sum() == 0: continue
            cmc += (np.cumsum(matches) > 0).astype(float)
            prec = np.cumsum(matches) / (np.arange(len(matches)) + 1)
            aps.append((prec * matches).sum() / matches.sum())
        cmc /= max(len(q_labels), 1)
        mAP = np.mean(aps) if aps else 0.0
        return cmc[0] * 100, mAP, total_loss / total_samples

    else:
        # 分类评估：MNIST/CIFAR10
        accs, losses = [], []
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
            if total > 0:
                accs.append(100. * correct / total)
                losses.append(running_loss / total)
        return np.mean(accs), np.mean(losses)


def run_proposed(config):
    train_loaders, test_loaders, num_classes = get_dataloaders(config)
    global_model = SharedBackbone(config['dataset']).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = []
    for i in range(config['num_sbs'] * config['clients_per_sbs']):
        modal = "RGB" if (i % 2 == 0) else "IR"
        clients.append(ClientUE(i, config, train_loaders[i], test_loaders[i], num_classes, modal))

    sbs_cache = {i: [] for i in range(config['num_sbs'])}
    is_reid = (config['dataset'] == 'SYSU-MM01')
    eval_every = config.get('eval_every', 1)
    history = {'R1': [], 'mAP': [], 'loss': []} if is_reid else {'accuracy': [], 'loss': []}
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

        # NaN 保护：如果聚合结果含 NaN，跳过本轮更新
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