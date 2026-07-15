import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def hetero_dirichlet_partition(targets, num_clients, num_classes, alpha):
    """使用狄利克雷分布进行Non-IID数据划分"""
    client_indices = [[] for _ in range(num_clients)]
    targets = np.array(targets)

    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        # 狄利克雷分布生成比例，累积分割点实现划分
        proportions = np.cumsum(np.random.dirichlet(np.repeat(alpha, num_clients))) * len(idx_k)
        splits = np.split(idx_k, proportions.astype(int)[:-1])
        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    return client_indices


def get_dataloaders(config):
    """加载数据集并返回客户端DataLoader、测试集DataLoader和类别数"""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 根据配置选择数据集
    if config['dataset'] == "CIFAR10":
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    targets = np.array(train_dataset.targets)

    # 两级Non-IID划分：SBS间 + SBS内
    sbs_indices = hetero_dirichlet_partition(targets, config['num_sbs'], 10, config['alpha_inter'])
    client_trainloaders = []

    for sbs_id in range(config['num_sbs']):
        sbs_targets = targets[sbs_indices[sbs_id]]
        intra_indices = hetero_dirichlet_partition(sbs_targets, config['clients_per_sbs'], 10, config['alpha_intra'])
        for client_idx_in_sbs in intra_indices:
            real_indices = np.array(sbs_indices[sbs_id])[client_idx_in_sbs]
            client_trainloaders.append(DataLoader(Subset(train_dataset, real_indices),
                                                  batch_size=config['batch_size'], shuffle=True))

    return client_trainloaders, DataLoader(test_dataset, batch_size=128, shuffle=False), 10