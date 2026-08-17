import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def split_data_dmfl(targets, num_sbs, clients_per_sbs, inner_client_iid, rng=None):
    """层级混合划分：edge间Non-IID，edge内按inner_client_iid混合。
    
    参考DMFL实现：
    1. 按标签排序后均匀切分给各edge（保证edge间Non-IID）
    2. Edge内：iid比例的IID数据 + (1-iid比例)的Non-IID数据混合
    
    Args:
        targets: 标签数组
        num_sbs: 边缘服务器数量
        clients_per_sbs: 每个SBS的客户端数量
        inner_client_iid: edge内IID数据比例 (0-1)
        rng: 随机数生成器
    
    Returns:
        user_map: {client_id: indices} 字典
    """
    if rng is None:
        rng = np.random.default_rng()
    
    num_users = num_sbs * clients_per_sbs
    u_per = clients_per_sbs
    
    labels = np.asarray(targets)
    idx = np.arange(len(labels))
    
    # 按标签排序后均匀切分给各edge（保证edge间Non-IID）
    sorted_idx = idx[labels.argsort()]
    shards = np.array_split(sorted_idx, num_sbs)
    
    user_map = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    for eid in range(num_sbs):
        shard = shards[eid]
        total = len(shard)
        n_iid = int(total * inner_client_iid)
        
        # IID池：随机采样
        iid_pool = shard[:n_iid].copy()
        rng.shuffle(iid_pool)
        iid_splits = np.array_split(iid_pool, u_per)
        
        # Non-IID池：按序分配
        niid_pool = shard[n_iid:]
        niid_splits = np.array_split(niid_pool, u_per)
        
        for j in range(u_per):
            uid = eid * u_per + j
            combined = np.concatenate([iid_splits[j], niid_splits[j]])
            rng.shuffle(combined)
            user_map[uid] = combined
    
    return user_map


def get_dataloaders(config):
    ds_name = config["dataset"]
    seed = int(config.get("seed", 42))
    np_rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if ds_name == "FashionMNIST":
        basic = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=basic)
        test_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=basic)
        num_classes = 10
    elif ds_name == "CIFAR10":
        basic = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=basic)
        test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=basic)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")

    targets = np.asarray(train_ds.targets)
    
    # 使用DMFL的层级混合划分
    inner_client_iid = config.get("inner_client_iid", 0.1)
    user_map = split_data_dmfl(
        targets, 
        config["num_sbs"], 
        config["clients_per_sbs"], 
        inner_client_iid, 
        np_rng
    )
    
    train_loaders = []
    loader_kwargs = {
        "num_workers": config.get("num_workers", 0),
        "pin_memory": config.get("device") == "cuda",
    }

    for client_id in range(config["num_sbs"] * config["clients_per_sbs"]):
        indices = user_map[client_id]
        if len(indices) == 0:
            raise RuntimeError(f"Client {client_id} has no data")
        
        train_subset = Subset(train_ds, indices)
        train_loader = DataLoader(
            train_subset, batch_size=config["batch_size"],
            shuffle=True, **loader_kwargs
        )
        train_loaders.append(train_loader)
    
    # 官方测试集（参考DMFL）
    test_loader = DataLoader(
        test_ds, batch_size=config.get("eval_batch_size", 128),
        shuffle=False, **loader_kwargs
    )
    
    return train_loaders, test_loader, num_classes
