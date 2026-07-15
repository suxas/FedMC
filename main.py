import numpy as np
import warnings
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders
from nodes.models import create_feature_extractor, create_classifier


def get_dataloaders_for_config(config):
    """根据配置选择数据集加载器"""
    if config['dataset'] == "SYSU-MM01":
        from utils.dataset_mm01 import get_dataloaders_sysumm01
        return get_dataloaders_sysumm01(config)
    else:
        return get_dataloaders(config)


def get_in_channels(dataset_name):
    """根据数据集名称获取输入通道数"""
    if dataset_name in ("CIFAR10", "SYSU-MM01"):
        return 3
    return 1


def get_global_state_dict(config, num_classes):
    """创建全局模型（feature_extractor + classifier）并返回合并的state_dict"""
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
    """根据数据集类型创建客户端列表"""
    total_clients = config['num_sbs'] * config['clients_per_sbs']

    if config['dataset'] == "SYSU-MM01":
        rgb_count = int(total_clients * config['modality_rgb_ratio'])
        modality_list = ['RGB'] * rgb_count + ['NIR'] * (total_clients - rgb_count)
        clients = [ClientUE(i, config, train_loaders[i], num_classes, modality=modality_list[i])
                   for i in range(total_clients)]
    else:
        clients = [ClientUE(i, config, train_loaders[i], num_classes)
                   for i in range(total_clients)]

    return clients


def dynamic_sparsify(grad, sparsity_ratio):
    """动态稀疏化：保留绝对值最大的(1-sparsity_ratio)比例梯度，其余置为NaN"""
    threshold = np.percentile(np.abs(grad), sparsity_ratio * 100)
    return np.where(np.abs(grad) >= threshold, grad, np.nan)


def low_rank_matrix_completion(mat, max_iter=5, tol=1e-4):
    """迭代式低秩矩阵补全(LRMC)：使用Soft-Impute算法，比单次SVD更精确

    Args:
        mat: 输入矩阵，NaN表示缺失
        max_iter: 最大迭代次数
        tol: 收敛阈值
    """
    if mat.size == 0 or np.isnan(mat).all():
        return mat

    nan_mask = np.where(np.isnan(mat))

    # 初始填充：列均值
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(mat, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)

    mat_filled = np.copy(mat)
    mat_filled[nan_mask] = np.take(col_mean, nan_mask[1])
    mat_filled = np.nan_to_num(mat_filled, nan=0.0)

    # 迭代式Soft-Impute
    for iteration in range(max_iter):
        mat_old = mat_filled.copy()

        # SVD降秩：基于能量比例自适应选择秩
        U, s, Vt = np.linalg.svd(mat_filled, full_matrices=False)
        cumulative_energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        rank = np.searchsorted(cumulative_energy, 0.9) + 1
        rank = max(1, min(rank, len(s)))
        s[rank:] = 0
        mat_reconstructed = U @ np.diag(s) @ Vt

        # 只更新缺失位置，保留已知值
        mat_filled[nan_mask] = mat_reconstructed[nan_mask]

        # 检查收敛
        if np.linalg.norm(mat_filled - mat_old) / (np.linalg.norm(mat_old) + 1e-10) < tol:
            break

    return mat_filled


def evaluate_global_model(config, global_state_dict, test_loader, device):
    """使用全局统一模型评估准确率和损失"""
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
    """将1D梯度向量应用到模型参数：state_dict -= global_lr * update_tensor"""
    idx = 0
    for name, param in global_state_dict.items():
        size = param.numel()
        update = torch.tensor(global_update_1d[idx:idx + size], dtype=param.dtype).reshape(param.shape).to(device)
        global_state_dict[name] -= global_lr * update
        idx += size


def run_proposed(config):
    """运行Proposed LRMC算法：动态稀疏化 + 迭代式低秩矩阵补全 + 多模态分组聚合 + 全局学习率衰减

    优化点：
    1. 迭代式LRMC（Soft-Impute）：比单次SVD恢复更精确
    2. 全局学习率衰减：后期减小更新步长，防止震荡
    3. 模态感知加权聚合：按客户端数据量加权，避免模态不平衡
    """
    train_loaders, test_loader, num_classes = get_dataloaders_for_config(config)
    global_state_dict = get_global_state_dict(config, num_classes)
    config['grad_dim'] = sum(p.numel() for p in global_state_dict.values())

    server = CloudServer(config)
    sbs_list = [EdgeSBS(i, config) for i in range(config['num_sbs'])]
    clients = create_clients(config, train_loaders, num_classes)

    # 记录每个客户端的数据量（用于加权聚合）
    client_data_sizes = [len(loader.dataset) for loader in train_loaders]
    total_data = sum(client_data_sizes)

    accuracy_history, loss_history = [], []
    global_rounds = config['global_rounds']
    pbar = tqdm(range(global_rounds), colour='blue', desc="[Proposed LRMC]")

    for round_idx in pbar:
        # 全局学习率衰减：后期减小更新步长
        global_lr = config.get('global_lr_init', 1.0) * (config.get('global_lr_decay', 0.999) ** round_idx)

        # 刷新所有客户端的信道质量和掉队状态
        for c in clients:
            c.refresh_state()

        edge_grads = []
        for sbs in sbs_list:
            start = sbs.id * config['clients_per_sbs']
            sbs_clients = clients[start:start + config['clients_per_sbs']]

            # 按模态分组收集梯度，记录数据量
            mat_list_rgb, mat_list_nir = [], []
            size_list_rgb, size_list_nir = [], []

            for client in sbs_clients:
                raw_grad = client.local_train(global_state_dict)
                client_data_size = client_data_sizes[client.id]

                # 信道质量达标才传输，否则置NaN
                if client.channel_quality >= config['noma_threshold']:
                    # 自适应稀疏率：掉队者多保留（低稀疏），正常者少保留（高稀疏）
                    sparsity = 0.1 if client.is_straggler_state else 0.5
                    processed_grad = dynamic_sparsify(raw_grad, sparsity)
                else:
                    processed_grad = np.full_like(raw_grad, np.nan)

                if client.modality == 'RGB':
                    mat_list_rgb.append(processed_grad)
                    size_list_rgb.append(client_data_size)
                else:
                    mat_list_nir.append(processed_grad)
                    size_list_nir.append(client_data_size)

            # 分模态进行迭代式LRMC恢复
            grad_dim = config['grad_dim']
            grad_rgb = np.array(mat_list_rgb) if mat_list_rgb else np.empty((0, grad_dim))
            grad_nir = np.array(mat_list_nir) if mat_list_nir else np.empty((0, grad_dim))
            completed_rgb = low_rank_matrix_completion(grad_rgb)
            completed_nir = low_rank_matrix_completion(grad_nir)

            # 模态感知加权聚合：按数据量加权
            valid = []
            weights = []
            if completed_rgb.size > 0:
                valid.append(np.nanmean(completed_rgb, axis=0))
                weights.append(sum(size_list_rgb))
            if completed_nir.size > 0:
                valid.append(np.nanmean(completed_nir, axis=0))
                weights.append(sum(size_list_nir))

            if valid:
                total_weight = sum(weights)
                weighted_grad = sum(w * g for w, g in zip(weights, valid)) / total_weight
                edge_grads.append(weighted_grad)
            else:
                edge_grads.append(np.zeros(grad_dim))

        # 全局聚合并更新模型参数（带全局学习率）
        global_update_1d = server.global_aggregate(edge_grads)
        apply_gradient_update(global_state_dict, global_update_1d, config['device'], global_lr)

        # 评估并记录
        mean_acc, mean_loss = evaluate_global_model(config, global_state_dict, test_loader, config['device'])
        accuracy_history.append(mean_acc)
        loss_history.append(mean_loss)
        pbar.set_postfix(Acc=f"{mean_acc:.2f}%", Loss=f"{mean_loss:.4f}", LR=f"{global_lr:.4f}")

    return {"accuracy": accuracy_history, "loss": loss_history}
