import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def hetero_dirichlet_partition(targets, num_clients, num_classes, alpha):
    """基于狄利克雷分布将数据集下标非IID地划分给多个客户端"""
    client_indices = [[] for _ in range(num_clients)]
    targets = np.array(targets)
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.cumsum(proportions) * len(idx_k)
        splits = np.split(idx_k, proportions.astype(int)[:-1])
        for i in range(num_clients):
            client_indices[i].extend(splits[i])
    return [np.array(idx) for idx in client_indices]


def get_dataloaders(config):
    if config['dataset'] == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    num_classes = 10
    targets = train_dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    # 第一级：按簇间狄利克雷分布划分给各 SBS
    sbs_indices = hetero_dirichlet_partition(targets, config['num_sbs'], num_classes, config['alpha_inter'])

    client_trainloaders = []
    for sbs_id in range(config['num_sbs']):
        sbs_targets = targets[sbs_indices[sbs_id]]
        # 第二级：SBS 内部再按簇内狄利克雷分布划分给各客户端
        intra_indices = hetero_dirichlet_partition(
            sbs_targets, config['clients_per_sbs'], num_classes, config['alpha_intra']
        )
        for client_idx_in_sbs in intra_indices:
            real_indices = sbs_indices[sbs_id][client_idx_in_sbs]
            subset = Subset(train_dataset, real_indices)
            loader = DataLoader(subset, batch_size=config['batch_size'], shuffle=True)
            client_trainloaders.append(loader)

    global_testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return client_trainloaders, global_testloader, num_classes
