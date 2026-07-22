"""数据加载：MNIST / CIFAR10 / SYSU-MM01"""

import os
import random
import numpy as np
from collections import defaultdict
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torch


# ============================================================
# MNIST / CIFAR10
# ============================================================

def _get_standard_dataset(dataset_name):
    if dataset_name == 'CIFAR10':
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
        return train_ds, test_ds, 10
    elif dataset_name == 'MNIST':
        tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), transforms.Resize((32, 32)),
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=tf)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=tf)
        return train_ds, test_ds, 10
    raise ValueError(f"不支持的数据集: {dataset_name}")


def split_data_inter_iid(dataset, num_users, num_edges, alpha_inter, alpha_intra):
    """两级IID划分：簇间alpha_inter IID + 簇内alpha_intra IID"""
    u_per = num_users // num_edges
    labels = np.array(dataset.targets)
    idx = np.arange(len(labels))
    num_classes = len(np.unique(labels))

    class_indices = {c: idx[labels == c] for c in range(num_classes)}

    sorted_classes = list(range(num_classes))
    np.random.shuffle(sorted_classes)
    class_shards = np.array_split(sorted_classes, num_edges)

    user_map = {i: np.array([], dtype='int64') for i in range(num_users)}

    for eid in range(num_edges):
        main_classes = class_shards[eid].tolist()
        other_classes = [c for c in range(num_classes) if c not in main_classes]

        inter_iid_size = 0
        if other_classes and alpha_inter > 0:
            inter_pool = np.concatenate([class_indices[c] for c in other_classes])
            inter_iid_size = int(len(inter_pool) * alpha_inter / num_edges)
            inter_pool = np.random.choice(inter_pool, size=inter_iid_size, replace=False)

        intra_pool = np.concatenate([class_indices[c] for c in main_classes])
        np.random.shuffle(intra_pool)

        n_iid = int(len(intra_pool) * alpha_intra)
        iid_pool = intra_pool[:n_iid]
        niid_pool = intra_pool[n_iid:]

        iid_splits = np.array_split(iid_pool, u_per)
        niid_splits = np.array_split(niid_pool, u_per)

        for j in range(u_per):
            uid = eid * u_per + j
            parts = [iid_splits[j]]
            if len(niid_splits) > j:
                parts.append(niid_splits[j])
            if inter_iid_size > 0:
                inter_splits = np.array_split(inter_pool, u_per)
                if len(inter_splits) > j:
                    parts.append(inter_splits[j])
            combined = np.concatenate(parts)
            np.random.shuffle(combined)
            user_map[uid] = combined

    return user_map


# ============================================================
# SYSU-MM01
# ============================================================

class SYSUMM01Dataset(Dataset):
    """SYSU-MM01：RGB（cam1/4/5）+ IR（cam2/3/6）双模态"""

    def __init__(self, root_dir, transform_rgb=None, transform_ir=None, id_list=None):
        self.transform_rgb = transform_rgb
        self.transform_ir = transform_ir
        rgb_cams = ['cam1', 'cam4', 'cam5']
        ir_cams = ['cam2', 'cam3', 'cam6']
        self.samples = []
        for cam in rgb_cams + ir_cams:
            cam_path = os.path.join(root_dir, cam)
            if not os.path.exists(cam_path):
                continue
            modality = 'RGB' if cam in rgb_cams else 'IR'
            for pid_str in sorted(os.listdir(cam_path)):
                pid = int(pid_str)
                if id_list is not None and pid not in id_list:
                    continue
                pid_path = os.path.join(cam_path, pid_str)
                if not os.path.isdir(pid_path):
                    continue
                for img_name in os.listdir(pid_path):
                    if img_name.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                        self.samples.append((os.path.join(pid_path, img_name), pid, modality))
        unique_pids = sorted(set(s[1] for s in self.samples))
        self.pid_map = {pid: idx for idx, pid in enumerate(unique_pids)}
        self.num_classes = len(unique_pids)
        self.pid_to_indices = defaultdict(lambda: {'RGB': [], 'IR': []})
        for idx, (_, pid, mod) in enumerate(self.samples):
            self.pid_to_indices[pid][mod].append(idx)
        self.pids = list(self.pid_to_indices.keys())
        self.targets = [self.pid_map[s[1]] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pid, modality = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if modality == 'RGB' and self.transform_rgb:
            img = self.transform_rgb(img)
        elif modality == 'IR' and self.transform_ir:
            img = self.transform_ir(img)
        return img, self.pid_map[pid], modality


class _ClientPKSampler:
    """单客户端PK采样器"""

    def __init__(self, dataset, client_indices, client_pids, P, K_rgb, K_ir):
        self.P = min(P, len(client_pids))
        self.K_rgb = K_rgb
        self.K_ir = K_ir
        self.batch_size = self.P * (K_rgb + K_ir)
        self.global_to_local = {g: l for l, g in enumerate(client_indices)}
        self.pid_to_local = {}
        for pid in client_pids:
            rgb_local = [self.global_to_local[i] for i in dataset.pid_to_indices[pid]['RGB']
                         if i in self.global_to_local]
            ir_local = [self.global_to_local[i] for i in dataset.pid_to_indices[pid]['IR']
                        if i in self.global_to_local]
            if rgb_local or ir_local:
                self.pid_to_local[pid] = {'RGB': rgb_local, 'IR': ir_local}
        self.valid_pids = list(self.pid_to_local.keys())
        self.n_batches = max(1, len(client_indices) // self.batch_size)

    def __iter__(self):
        for _ in range(self.n_batches):
            selected_pids = random.choices(self.valid_pids, k=self.P)
            batch_indices = []
            for pid in selected_pids:
                pools = self.pid_to_local[pid]
                rgb_pool, ir_pool = pools['RGB'], pools['IR']
                if len(rgb_pool) >= self.K_rgb:
                    batch_indices.extend(random.sample(rgb_pool, self.K_rgb))
                elif rgb_pool:
                    batch_indices.extend(random.choices(rgb_pool, k=self.K_rgb))
                if len(ir_pool) >= self.K_ir:
                    batch_indices.extend(random.sample(ir_pool, self.K_ir))
                elif ir_pool:
                    batch_indices.extend(random.choices(ir_pool, k=self.K_ir))
            if batch_indices:
                random.shuffle(batch_indices)
                yield batch_indices

    def __len__(self):
        return self.n_batches


def _reid_collate_fn(batch):
    images, labels, modalities = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(modalities)


def _get_train_test_ids(root_dir):
    exp_dir = os.path.join(root_dir, 'exp')
    train_file = os.path.join(exp_dir, 'train_id.txt')
    test_file = os.path.join(exp_dir, 'test_id.txt')

    def parse_ids(filepath):
        if os.path.exists(filepath):
            text = open(filepath).read().replace('\n', ',')
            return [int(x.strip()) for x in text.split(',') if x.strip()]
        all_ids = list(range(1, 492))
        np.random.seed(42)
        np.random.shuffle(all_ids)
        return all_ids[:296] if 'train' in filepath else all_ids[296:]

    return parse_ids(train_file), parse_ids(test_file)


def _get_sysumm01_loaders(config):
    root_dir = config.get('data_path', './data/SYSU-MM01')
    input_size = config.get('input_size', 256)
    P = config.get('pk_P', 8)
    K = config.get('pk_K', 4)
    batch_size = config.get('batch_size', 64)
    total_clients = config['num_sbs'] * config['clients_per_sbs']
    num_edges = config['num_sbs']
    alpha_inter = config.get('alpha_inter', 0.5)

    train_ids, test_ids = _get_train_test_ids(root_dir)

    train_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(input_size, padding=16),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = SYSUMM01Dataset(root_dir, train_tf, train_tf, train_ids)
    test_ds = SYSUMM01Dataset(root_dir, test_tf, test_tf, test_ids)
    num_classes = train_ds.num_classes

    u_per = total_clients // num_edges
    sorted_pids = np.array(sorted(train_ds.pids))
    num_pids = len(sorted_pids)

    np.random.shuffle(sorted_pids)
    n_inter_iid = int(num_pids * alpha_inter)
    inter_iid_pids = sorted_pids[:n_inter_iid]
    remaining_pids = sorted_pids[n_inter_iid:]
    niid_shards = np.array_split(remaining_pids, num_edges)

    client_loaders = []
    for eid in range(num_edges):
        main_pids = niid_shards[eid].tolist()
        inter_pids = inter_iid_pids.tolist()
        all_cluster_pids = main_pids + inter_pids

        np.random.shuffle(all_cluster_pids)
        client_pid_list = np.array_split(all_cluster_pids, u_per)

        for j in range(u_per):
            client_pids = set(client_pid_list[j].tolist())
            client_indices = [idx for idx, (_, pid, _) in enumerate(train_ds.samples)
                              if pid in client_pids]
            client_subset = Subset(train_ds, client_indices)
            pk_sampler = _ClientPKSampler(train_ds, client_indices, client_pids, P, K, K)
            client_loaders.append(DataLoader(client_subset, batch_sampler=pk_sampler,
                                             collate_fn=_reid_collate_fn, num_workers=0,
                                             pin_memory=True))

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=_reid_collate_fn, num_workers=0, pin_memory=True)
    return client_loaders, test_loader, num_classes


# ============================================================
# 统一入口
# ============================================================

def get_dataloaders(config):
    dataset_name = config['dataset']
    if dataset_name == 'SYSU-MM01':
        return _get_sysumm01_loaders(config)

    train_ds, test_ds, num_classes = _get_standard_dataset(dataset_name)
    groups = split_data_inter_iid(train_ds,
                                  config['num_sbs'] * config['clients_per_sbs'],
                                  config['num_sbs'],
                                  config.get('alpha_inter', 0.5),
                                  config.get('alpha_intra', 1.0))

    client_loaders = []
    for i in range(len(groups)):
        client_loaders.append(DataLoader(Subset(train_ds, list(groups[i])),
                                         batch_size=config['batch_size'], shuffle=True))
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    return client_loaders, test_loader, num_classes
