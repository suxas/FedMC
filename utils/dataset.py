import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class SYSUMM01Dataset(Dataset):
    """SYSU-MM01 行人重识别数据集，返回 (RGB图, IR图, 标签)"""

    def __init__(self, data_dir='./data/SYSU-MM01', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.rgb_cams = ['cam1', 'cam2', 'cam4', 'cam5']
        self.ir_cams = ['cam3', 'cam6']
        self.rgb_paths, self.ir_paths = {}, {}

        for cam in self.rgb_cams:
            cam_dir = os.path.join(self.data_dir, cam)
            if not os.path.exists(cam_dir): continue
            for pid in os.listdir(cam_dir):
                pid_dir = os.path.join(cam_dir, pid)
                if os.path.isdir(pid_dir):
                    if pid not in self.rgb_paths: self.rgb_paths[pid] = []
                    for img in os.listdir(pid_dir):
                        if img.endswith('.jpg'):
                            self.rgb_paths[pid].append(os.path.join(pid_dir, img))

        for cam in self.ir_cams:
            cam_dir = os.path.join(self.data_dir, cam)
            if not os.path.exists(cam_dir): continue
            for pid in os.listdir(cam_dir):
                pid_dir = os.path.join(cam_dir, pid)
                if os.path.isdir(pid_dir):
                    if pid not in self.ir_paths: self.ir_paths[pid] = []
                    for img in os.listdir(pid_dir):
                        if img.endswith('.jpg'):
                            self.ir_paths[pid].append(os.path.join(pid_dir, img))

        self.identities = [pid for pid in self.rgb_paths.keys()
                           if pid in self.ir_paths and len(self.rgb_paths[pid]) > 0 and len(self.ir_paths[pid]) > 0]
        self.identities.sort()

        if len(self.identities) == 0:
            raise RuntimeError(f"未在 {data_dir} 中找到有效的成对 RGB-IR 图像！")

        self.id_to_label = {pid: i for i, pid in enumerate(self.identities)}
        self.samples = []
        for pid in self.identities:
            for path in self.rgb_paths[pid]:
                self.samples.append({'pid': pid, 'rgb_path': path})

        self.targets = np.array([self.id_to_label[s['pid']] for s in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample['pid']
        rgb_path = sample['rgb_path']
        ir_path = random.choice(self.ir_paths[pid])

        rgb_img = Image.open(rgb_path).convert('RGB')
        ir_img = Image.open(ir_path).convert('L')

        if self.transform:
            rgb_img = self.transform(rgb_img)
            ir_img = self.transform(ir_img)

        label = self.id_to_label[pid]
        return rgb_img, ir_img, label


class DualModalWrapper(Dataset):
    """将单模态数据集包装为双模态格式 (img, img, label)"""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, img, label


def hetero_dirichlet_partition(targets, num_clients, num_classes, alpha):
    """Dirichlet 分布异构划分（向量化版本，适用于大量类别）"""
    targets = np.array(targets)
    client_bufs = [[] for _ in range(num_clients)]

    # 一次排序，按类别分组索引
    sorted_idx = np.argsort(targets, kind='stable')
    class_counts = np.bincount(targets, minlength=num_classes)
    offsets = np.zeros(num_classes + 1, dtype=int)
    np.cumsum(class_counts, out=offsets[1:])

    for k in range(num_classes):
        n_k = offsets[k + 1] - offsets[k]
        if n_k == 0:
            continue
        idx_k = sorted_idx[offsets[k]:offsets[k + 1]]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions[:-1]) * n_k).astype(int)
        splits = np.split(idx_k, split_points)
        for i, s in enumerate(splits):
            if len(s) > 0:
                client_bufs[i].append(s)

    return [np.concatenate(c) if c else np.array([], dtype=int) for c in client_bufs]


def assign_modal_types(num_clients_per_sbs, modal_split_ratio):
    """为每个 SBS 簇内的 UAV 分配模态类型

    Returns:
        list of str: ["RGB", "IR", "Both", ...]
    """
    ratio = modal_split_ratio
    n_rgb = int(num_clients_per_sbs * ratio.get("RGB", 0.4))
    n_ir = int(num_clients_per_sbs * ratio.get("IR", 0.4))
    n_both = num_clients_per_sbs - n_rgb - n_ir

    modal_list = ["RGB"] * n_rgb + ["IR"] * n_ir + ["Both"] * n_both
    random.shuffle(modal_list)
    return modal_list


def get_dataloaders(config):
    """获取数据加载器，支持模态异构划分

    Returns:
        train_loaders: list of DataLoader (每个客户端一个)
        test_loaders: list of DataLoader (每个客户端一个)
        num_classes: 类别数
        modal_types: list of str (每个客户端的模态类型)
    """
    ds_name = config['dataset']
    sysu_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    if ds_name == "MNIST":
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=basic_transform)
        train_ds = DualModalWrapper(train_ds)
        num_classes = 10
    elif ds_name == "CIFAR10":
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=basic_transform)
        train_ds = DualModalWrapper(train_ds)
        num_classes = 10
    else:
        train_ds = SYSUMM01Dataset(data_dir='./data/SYSU-MM01', transform=sysu_transform)
        num_classes = len(train_ds.identities)
        print(f"[*] SYSU-MM01: 找到 {num_classes} 个身份，共 {len(train_ds)} 样本。")

    targets = np.array([train_ds[i][2] for i in range(len(train_ds))]) if ds_name != "SYSUMM01" else train_ds.targets

    # 簇间 Dirichlet 划分
    print(f"[*] 簇间划分: {config['num_sbs']} 个SBS, alpha={config['alpha_inter']}")
    sbs_indices = hetero_dirichlet_partition(targets, config['num_sbs'], num_classes, config['alpha_inter'])

    client_trainloaders, client_testloaders, modal_types = [], [], []
    modal_split_ratio = config.get('modal_split_ratio', {"RGB": 0.4, "IR": 0.4, "Both": 0.2})

    for sbs_id in range(config['num_sbs']):
        print(f"[*] SBS {sbs_id}: 簇内划分 {config['clients_per_sbs']} 个客户端...")
        sbs_targets = targets[sbs_indices[sbs_id]]
        # 簇内 Dirichlet 划分
        intra_indices = hetero_dirichlet_partition(sbs_targets, config['clients_per_sbs'], num_classes,
                                                   config['alpha_intra'])

        # 为簇内 UAV 分配模态类型
        sbs_modal_types = assign_modal_types(config['clients_per_sbs'], modal_split_ratio)

        for client_idx_in_sbs, client_modal in enumerate(sbs_modal_types):
            real_indices = sbs_indices[sbs_id][intra_indices[client_idx_in_sbs]]
            np.random.shuffle(real_indices)

            split_point = max(1, int(len(real_indices) * 0.8))
            train_idx, test_idx = real_indices[:split_point], real_indices[split_point:]
            if len(test_idx) == 0: test_idx = train_idx

            num_workers = config.get('num_workers', 0)
            use_pin = config['device'] == 'cuda'
            use_persistent = num_workers > 0

            client_trainloaders.append(DataLoader(
                Subset(train_ds, train_idx), batch_size=config['batch_size'], shuffle=True,
                num_workers=num_workers, pin_memory=use_pin, persistent_workers=use_persistent
            ))
            client_testloaders.append(DataLoader(
                Subset(train_ds, test_idx), batch_size=128, shuffle=False,
                num_workers=num_workers, pin_memory=use_pin, persistent_workers=use_persistent
            ))
            modal_types.append(client_modal)

    print(f"[*] 数据划分完成: {len(client_trainloaders)} 个客户端, {num_classes} 个类别")
    return client_trainloaders, client_testloaders, num_classes, modal_types
