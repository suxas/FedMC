import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
CIFAR100_COARSE_CLASSES = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
    'household electrical devices', 'household furniture', 'insects', 'large carnivores',
    'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
    'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
    'trees', 'vehicles 1', 'vehicles 2'
]

CIFAR100_FINE_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


class MultimodalCIFAR100(Dataset):
    """Multimodal-CIFAR100: 图像模态 + 文本 Prompt 模态"""
    def __init__(self, root="./data", train=True, transform=None, seq_len=16):
        self.cifar = datasets.CIFAR100(root=root, train=train, download=True)
        self.transform = transform
        self.seq_len = seq_len
        self.targets = self.cifar.targets

        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for name in CIFAR100_FINE_CLASSES + CIFAR100_COARSE_CLASSES:
            for word in name.replace('_', ' ').split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

        self.class_text_tokens = {}
        for fine_idx, fine_name in enumerate(CIFAR100_FINE_CLASSES):
            coarse_idx = fine_idx // 5
            coarse_name = CIFAR100_COARSE_CLASSES[coarse_idx]
            desc = f"a photo of {fine_name.replace('_', ' ')} a type of {coarse_name}"
            tokens = [self.vocab.get(w, 1) for w in desc.split()][:seq_len]
            if len(tokens) < seq_len:
                tokens = tokens + [0] * (seq_len - len(tokens))
            self.class_text_tokens[fine_idx] = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, target = self.cifar[idx]
        if self.transform is not None:
            img = self.transform(img)
        text_tokens = self.class_text_tokens[target]
        return (img, text_tokens), target


def split_data_dmfl(targets, num_sbs, clients_per_sbs, inner_client_iid, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    num_users = num_sbs * clients_per_sbs
    u_per = clients_per_sbs

    labels = np.asarray(targets)
    idx = np.arange(len(labels))

    sorted_idx = idx[labels.argsort()]
    shards = np.array_split(sorted_idx, num_sbs)
    user_map = {i: np.array([], dtype='int64') for i in range(num_users)}

    for eid in range(num_sbs):
        shard = shards[eid]
        total = len(shard)
        n_iid = int(total * inner_client_iid)

        iid_pool = shard[:n_iid].copy()
        rng.shuffle(iid_pool)
        iid_splits = np.array_split(iid_pool, u_per)

        niid_pool = shard[n_iid:]
        niid_splits = np.array_split(niid_pool, u_per)

        for j in range(u_per):
            uid = eid * u_per + j
            combined = np.concatenate([iid_splits[j], niid_splits[j]])
            rng.shuffle(combined)
            user_map[uid] = combined

    return user_map


def get_dataloaders(config):
    seed = int(config.get("seed", 42))
    np_rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])

    seq_len = config.get("text_seq_len", 16)
    train_ds = MultimodalCIFAR100(root="./data", train=True, transform=train_transform, seq_len=seq_len)
    test_ds = MultimodalCIFAR100(root="./data", train=False, transform=test_transform, seq_len=seq_len)
    num_classes = 100

    targets = np.asarray(train_ds.targets)
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
        train_subset = Subset(train_ds, indices)
        train_loader = DataLoader(
            train_subset, batch_size=config["batch_size"],
            shuffle=True, **loader_kwargs
        )
        train_loaders.append(train_loader)

    test_loader = DataLoader(
        test_ds, batch_size=config.get("eval_batch_size", 128),
        shuffle=False, **loader_kwargs
    )

    return train_loaders, test_loader, num_classes