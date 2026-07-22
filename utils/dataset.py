import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class SYSUMM01Dataset(Dataset):
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
    def __init__(self, base_dataset): self.base = base_dataset

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, img, label


def hetero_dirichlet_partition(targets, num_clients, num_classes, alpha):
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

    targets = np.array([train_ds[i][2] for i in range(len(train_ds))]) if ds_name != "SYSU-MM01" else train_ds.targets
    sbs_indices = hetero_dirichlet_partition(targets, config['num_sbs'], num_classes, config['alpha_inter'])
    client_trainloaders, client_testloaders = [], []

    for sbs_id in range(config['num_sbs']):
        sbs_targets = targets[sbs_indices[sbs_id]]
        intra_indices = hetero_dirichlet_partition(sbs_targets, config['clients_per_sbs'], num_classes,
                                                   config['alpha_intra'])
        for client_idx_in_sbs in intra_indices:
            real_indices = sbs_indices[sbs_id][client_idx_in_sbs]
            np.random.shuffle(real_indices)

            split_point = max(1, int(len(real_indices) * 0.8))
            train_idx, test_idx = real_indices[:split_point], real_indices[split_point:]
            if len(test_idx) == 0: test_idx = train_idx

            # 恢复简单的单线程加载
            client_trainloaders.append(DataLoader(
                Subset(train_ds, train_idx), batch_size=config['batch_size'], shuffle=True
            ))
            client_testloaders.append(DataLoader(
                Subset(train_ds, test_idx), batch_size=128, shuffle=False
            ))

    return client_trainloaders, client_testloaders, num_classes