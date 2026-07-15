import numpy as np
import warnings
import torch
from tqdm import tqdm
from nodes.clients import ClientUAV
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders
from nodes.models import create_feature_extractor, create_classifier


def get_in_channels(dataset_name):
    """所有数据集统一3通道（MNIST在数据加载时转为3通道）"""
    return 3


def get_global_state_dict(config, num_classes):
    """创建全局模型并返回合并的state_dict"""
    dataset = config['dataset']
    in_channels = get_in_channels(dataset)

    fe = create_feature_extractor(dataset, in_channels).to(config['device'])
    feature_dim = fe.get_feature_dim()
    cl = create_classifier(dataset, feature_dim, num_classes).to(config['device'])

    global_sd = {}
    for k, v in fe.named_parameters():
        global_sd[f'feature_extractor.{k}'] = v.data
    for k, v in cl.named_parameters():
        global_sd[f'classifier.{k}'] = v.data
    return global_sd


def create_clients(config, train_loaders, num_classes):
    """创建UAV客户端列表"""
    total_clients = config['num_sbs'] * config['clients_per_sbs']
    clients = [ClientUAV(i, config, train_loaders[i], num_classes)
               for i in range(total_clients)]
    return clients


def dynamic_sparsify(grad, sparsity_ratio):
    """动态稀疏化：保留绝对值最大的(1-sparsity_ratio)比例梯度，其余置为NaN"""
    threshold = np.percentile(np.abs(grad), sparsity_ratio * 100)
    return np.where(np.abs(grad) >= threshold, grad, np.nan)


def low_rank_matrix_completion(mat, max_iter=5, tol=1e-4):
    """迭代式低秩矩阵补全(LRMC)：Soft-Impute算法"""
    if mat.size == 0 or np.isnan(mat).all():
        return mat

    nan_mask = np.where(np.isnan(mat))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(mat, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)

    mat_filled = np.copy(mat)
    mat_filled[nan_mask] = np.take(col_mean, nan_mask[1])
    mat_filled = np.nan_to_num(mat_filled, nan=0.0)

    for iteration in range(max_iter):
        mat_old = mat_filled.copy()

        U, s, Vt = np.linalg.svd(mat_filled, full_matrices=False)
        cumulative_energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        rank = np.searchsorted(cumulative_energy, 0.9) + 1
        rank = max(1, min(rank, len(s)))
        s[rank:] = 0
        mat_reconstructed = U @ np.diag(s) @ Vt

        mat_filled[nan_mask] = mat_reconstructed[nan_mask]

        if np.linalg.norm(mat_filled - mat_old) / (np.linalg.norm(mat_old) + 1e-10) < tol:
            break

    return mat_filled


def evaluate_global_model(config, global_state_dict, test_loader, device):
    """使用全局模型评估准确率和损失"""
    dataset = config['dataset']
    in_channels = get_in_channels(dataset)

    fe = create_feature_extractor(dataset, in_channels).to(device)
    fe_keys = [k for k in global_state_dict if k.startswith('feature_extractor.')]
    fe_sd = {k.replace('feature_extractor.', ''): global_state_dict[k] for k in fe_keys}
    fe.load_state_dict(fe_sd, strict=False)

    feature_dim = fe.get_feature_dim()
    cl_keys = [k for k in global_state_dict if k.startswith('classifier.')]
    num_classes = global_state_dict[[k for k in cl_keys if 'fc' in k and 'weight' in k][-1]].shape[0]
    cl = create_classifier(dataset, feature_dim, num_classes).to(device)
    cl_sd = {k.replace('classifier.', ''): global_state_dict[k] for k in cl_keys}
    cl.load_state_dict(cl_sd, strict=False)

    fe.eval()
    cl.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cl(fe(images))
            running_loss += criterion(outputs, labels).item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total, running_loss / total


def apply_gradient_update(global_state_dict, global_update_1d, device, global_lr=1.0):
    """将1D梯度向量应用到模型参数"""
    idx = 0
    for name, param in global_state_dict.items():
        size = param.numel()
        update = torch.tensor(global_update_1d[idx:idx + size], dtype=param.dtype).reshape(param.shape).to(device)
        global_state_dict[name] -= global_lr * update
        idx += size


def run_proposed(config):
    """运行Proposed算法：空中计算 + 动态稀疏化 + LRMC梯度补全 + 全局学习率衰减

    流程：
    1. 各UAV本地训练
    2. 簇内：掉队者梯度置NaN，正常者动态稀疏化，SBS用空中计算叠加接收
    3. SBS对簇内梯度矩阵做LRMC补全，恢复丢失分量
    4. 簇间：LBS常规聚合各SBS的梯度
    """
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    global_state_dict = get_global_state_dict(config, num_classes)
    config['grad_dim'] = sum(p.numel() for p in global_state_dict.values())

    server = CloudServer(config)
    sbs_list = [EdgeSBS(i, config) for i in range(config['num_sbs'])]
    clients = create_clients(config, train_loaders, num_classes)

    client_data_sizes = [len(loader.dataset) for loader in train_loaders]

    accuracy_history, loss_history = [], []
    global_rounds = config['global_rounds']
    pbar = tqdm(range(global_rounds), colour='blue', desc="[Proposed Air+LRMC]")

    for round_idx in pbar:
        global_lr = config.get('global_lr_init', 1.0) * (config.get('global_lr_decay', 0.999) ** round_idx)

        for c in clients:
            c.refresh_state(round_idx)

        edge_grads = []
        for sbs in sbs_list:
            start = sbs.id * config['clients_per_sbs']
            sbs_clients = clients[start:start + config['clients_per_sbs']]

            # 收集簇内各UAV的梯度
            grad_matrix = []
            uav_weights = []

            for client in sbs_clients:
                raw_grad = client.local_train(global_state_dict)
                client_weight = client_data_sizes[client.id]

                if client.is_straggler_state:
                    processed_grad = np.full_like(raw_grad, np.nan)
                else:
                    sparsity = 0.1 if client.channel_quality < 0.5 else 0.5
                    processed_grad = dynamic_sparsify(raw_grad, sparsity)

                grad_matrix.append(processed_grad)
                uav_weights.append(client_weight)

            # 簇内：LRMC补全 + 空中计算聚合
            grad_array = np.array(grad_matrix)
            completed = low_rank_matrix_completion(grad_array)

            cluster_grad = sbs.air_computing_aggregate(
                [completed[i] for i in range(len(completed))],
                uav_weights
            )
            edge_grads.append(cluster_grad)

        # 簇间：LBS常规聚合各SBS梯度
        global_update_1d = server.global_aggregate(edge_grads)
        apply_gradient_update(global_state_dict, global_update_1d, config['device'], global_lr)

        mean_acc, mean_loss = evaluate_global_model(config, global_state_dict, test_loader, config['device'])
        accuracy_history.append(mean_acc)
        loss_history.append(mean_loss)
        pbar.set_postfix(Acc=f"{mean_acc:.2f}%", Loss=f"{mean_loss:.4f}", LR=f"{global_lr:.4f}")

    return {"accuracy": accuracy_history, "loss": loss_history}
