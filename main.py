import numpy as np
import warnings
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders
from nodes.models import FeatureExtractor


def dynamic_sparsify(grad, sparsity_ratio):
    threshold = np.percentile(np.abs(grad), sparsity_ratio * 100)
    return np.where(np.abs(grad) >= threshold, grad, np.nan)


def low_rank_matrix_completion(mat):
    if np.isnan(mat).all():
        return np.zeros_like(mat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(mat, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    inds = np.where(np.isnan(mat))

    mat_filled = np.copy(mat)
    mat_filled[inds] = np.take(col_mean, inds[1])
    mat_filled = np.nan_to_num(mat_filled, nan=0.0)

    U, s, Vt = np.linalg.svd(mat_filled, full_matrices=False)
    rank = max(1, int(len(s) * 0.2))
    s[rank:] = 0
    mat_reconstructed = np.dot(U, np.dot(np.diag(s), Vt))

    mat_final = np.copy(mat)
    mat_final[inds] = mat_reconstructed[inds]
    return mat_final


def evaluate_global_model(clients, global_state_dict, test_loader, device):
    """通过每个客户端的个性化头评估全局特征，取平均以评估全局系统性能"""
    accs, losses = [], []
    criterion = torch.nn.CrossEntropyLoss()

    for client in clients:
        client.set_global_features(global_state_dict)
        client.feature_extractor.eval()
        client.classifier.eval()

        correct, total, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = client.feature_extractor(images)
                outputs = client.classifier(features)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accs.append(100. * correct / total)
        losses.append(running_loss / total)

    return np.mean(accs), np.mean(losses)


def run_proposed(config):
    # 1. 挂载数据与模型
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

    # 2. 绿色进度条
    pbar = tqdm(range(config['global_rounds']), colour='green', desc="[Training Proposed(LRMC+NOMA)]")

    for round_idx in pbar:
        for c in clients: c.refresh_state()
        edge_grads = []

        for sbs in sbs_list:
            cluster_clients = clients[sbs.id * config['clients_per_sbs']: (sbs.id + 1) * config['clients_per_sbs']]
            mat_list = []

            for client in cluster_clients:
                # 真实获取 1D 梯度向量
                raw_grad = client.local_train(global_state_dict)

                if client.channel_quality >= config['noma_threshold']:
                    sparsity = 0.1 if client.is_straggler_state else 0.6
                    mat_list.append(dynamic_sparsify(raw_grad, sparsity_ratio=sparsity))
                else:
                    mat_list.append(np.full_like(raw_grad, np.nan))

            grad_matrix = np.array(mat_list)
            completed_matrix = low_rank_matrix_completion(grad_matrix)

            edge_grad = np.nanmean(completed_matrix, axis=0)
            edge_grads.append(edge_grad)

        # 3. MBS 全局聚合与参数还原
        global_update_1d = server.global_aggregate(edge_grads)

        # 将 1D Numpy 还原回 PyTorch 权重字典
        idx = 0
        for name, param in global_state_dict.items():
            size = param.numel()
            update_tensor = torch.tensor(global_update_1d[idx:idx + size]).reshape(param.shape).to(config['device'])
            global_state_dict[name] -= update_tensor  # 更新权重
            idx += size

        global_model.load_state_dict(global_state_dict)

        # 4. 真实评估
        mean_acc, mean_loss = evaluate_global_model(clients, global_state_dict, test_loader, config['device'])
        accuracy_history.append(mean_acc)

        pbar.set_postfix(Acc=f"{mean_acc:.2f}%", Loss=f"{mean_loss:.4f}")

    return accuracy_history