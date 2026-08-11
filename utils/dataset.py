import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def hetero_dirichlet_partition(targets, num_clients, num_classes, alpha, rng=None):
    """Partition samples by class while keeping the result finite and stable."""
    rng = np.random.default_rng() if rng is None else rng
    targets = np.asarray(targets)
    client_bufs = [[] for _ in range(num_clients)]
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(targets == class_id)
        if len(class_indices) == 0:
            continue
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, max(alpha, 1e-3)))
        split_points = (np.cumsum(proportions[:-1]) * len(class_indices)).astype(int)
        for client_id, indices in enumerate(np.split(class_indices, split_points)):
            if len(indices):
                client_bufs[client_id].append(indices)
    return [np.concatenate(parts) if parts else np.empty(0, dtype=int) for parts in client_bufs]


def _rebalance_empty(partitions, minimum_size=2):
    """Move samples from the largest partitions to prevent invalid loaders."""
    partitions = [np.asarray(part).copy() for part in partitions]
    for index, partition in enumerate(partitions):
        if len(partition) >= minimum_size:
            continue
        donor = max((i for i in range(len(partitions)) if i != index),
                    key=lambda i: len(partitions[i]), default=None)
        if donor is None or len(partitions[donor]) <= minimum_size:
            continue
        needed = minimum_size - len(partition)
        take = min(needed, len(partitions[donor]) - minimum_size)
        partitions[index] = np.concatenate((partition, partitions[donor][:take]))
        partitions[donor] = partitions[donor][take:]
    return partitions


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
        num_classes = 10
    elif ds_name == "CIFAR10":
        basic = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=basic)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")

    targets = np.asarray(train_ds.targets)
    sbs_indices = hetero_dirichlet_partition(
        targets, config["num_sbs"], num_classes, config["alpha_inter"], np_rng
    )
    sbs_indices = _rebalance_empty(sbs_indices, minimum_size=2)
    train_loaders, test_loaders = [], []
    loader_kwargs = {
        "num_workers": config.get("num_workers", 0),
        "pin_memory": config.get("device") == "cuda",
    }

    for sbs_id in range(config["num_sbs"]):
        sbs_targets = targets[sbs_indices[sbs_id]]
        intra = hetero_dirichlet_partition(
            sbs_targets, config["clients_per_sbs"], num_classes,
            config["alpha_intra"], np_rng
        )
        intra = _rebalance_empty(intra, minimum_size=2)
        for client_index in range(config["clients_per_sbs"]):
            indices = sbs_indices[sbs_id][intra[client_index]] if len(intra[client_index]) else np.empty(0, dtype=int)
            if len(indices) == 0:
                raise RuntimeError(
                    "Dirichlet partition produced an empty client; increase alpha or reduce clients_per_sbs"
                )
            np_rng.shuffle(indices)
            split = max(1, int(len(indices) * 0.8))
            train_idx = indices[:split]
            test_idx = indices[split:] if len(indices) > split else indices[-1:]
            train_subset = Subset(train_ds, train_idx)
            train_loader = DataLoader(
                train_subset, batch_size=config["batch_size"],
                shuffle=len(train_idx) > 0, **loader_kwargs
            )
            test_loader = DataLoader(
                Subset(train_ds, test_idx), batch_size=config.get("eval_batch_size", 128),
                shuffle=False, **loader_kwargs
            )
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
    return train_loaders, test_loaders, num_classes
