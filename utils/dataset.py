import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def split_classes_disjoint(num_classes, num_clusters):
    """将类别无重叠地分配给各簇，返回 {cluster_id: [class_ids]}"""
    classes_per_cluster = num_classes // num_clusters
    remainder = num_classes % num_clusters
    assignment = {}
    start = 0
    for i in range(num_clusters):
        end = start + classes_per_cluster + (1 if i < remainder else 0)
        assignment[i] = list(range(start, end))
        start = end
    return assignment


def intra_cluster_partition(cluster_indices, targets, num_clients, alpha_intra):
    """簇内数据划分

    alpha_intra 表示簇内数据重叠比例：
      0   = 完全不重叠，每个UAV的数据互不相交
      0.1 = 10%的数据在簇内所有UAV间共享（IID部分）
      1   = 100%共享，所有UAV拿到完全相同的数据

    实现：
      - 共享池大小 = alpha_intra × 总数据量，所有UAV共享
      - 剩余部分均分给各UAV（不重叠）
    """
    n = len(cluster_indices)
    targets = np.array(targets)

    shared_count = int(n * alpha_intra)
    perm = np.random.permutation(cluster_indices)
    shared_pool = perm[:shared_count]
    unique_pool = perm[shared_count:]

    # 唯一部分均分给各UAV
    unique_per_client = len(unique_pool) // num_clients
    client_indices = []
    for i in range(num_clients):
        start = i * unique_per_client
        end = start + unique_per_client if i < num_clients - 1 else len(unique_pool)
        client_unique = unique_pool[start:end]
        # 共享池 + 唯一部分 = 该UAV的完整数据
        client_indices.append(np.concatenate([shared_pool, client_unique]).tolist())

    return client_indices


def get_dataloaders(config):
    """加载数据集并返回客户端DataLoader、测试集DataLoader和类别数

    数据划分策略：
      - 簇间：类别完全无重叠（严格Non-IID）
      - 簇内：alpha_intra控制数据重叠比例
    """
    dataset_name = config['dataset']

    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError(f"未知数据集: {dataset_name}, 可选: MNIST, CIFAR10, CIFAR100")

    targets = np.array(train_dataset.targets)

    # 簇间：类别完全无重叠
    class_assignment = split_classes_disjoint(num_classes, config['num_sbs'])
    all_indices = np.arange(len(targets))

    client_trainloaders = []
    for sbs_id in range(config['num_sbs']):
        cluster_classes = class_assignment[sbs_id]
        # 该簇拥有的数据：只包含分配给该簇的类别
        cluster_mask = np.isin(targets, cluster_classes)
        cluster_indices = all_indices[cluster_mask]

        # 簇内划分
        intra_partitions = intra_cluster_partition(
            cluster_indices, targets, config['clients_per_sbs'], config['alpha_intra']
        )

        for client_idx_list in intra_partitions:
            client_trainloaders.append(DataLoader(Subset(train_dataset, client_idx_list),
                                                  batch_size=config['batch_size'], shuffle=True))

    return client_trainloaders, DataLoader(test_dataset, batch_size=128, shuffle=False), num_classes
