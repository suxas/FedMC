import os
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

# ==============================================================================
# 1. CIFAR-100 多模态构建 (保持原样)
# ==============================================================================
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
    """Multimodal-CIFAR100: 图像模态 (32x32) + 文本 Prompt 模态"""
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


# ==============================================================================
# 2. TinyImageNet-200 真实多模态构建 (多线程+二进制缓存秒级加载)
# ==============================================================================
def _read_single_image(args):
    idx, path = args
    with Image.open(path) as img:
        return idx, np.asarray(img.convert("RGB"))


class MultimodalTinyImageNet(Dataset):
    """Multimodal-TinyImageNet: 真实图像 (64x64, 200类) + WordNet 语义 (高速缓存版)"""
    TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def __init__(self, root="./data", train=True, transform=None, seq_len=24, vocab_size=1000):
        self.root = root
        self.train = train
        self.transform = transform
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dataset_dir = os.path.join(self.root, "tiny-imagenet-200")

        self._ensure_dataset_exists()

        # 1. 解析 wnids.txt
        wnids_path = os.path.join(self.dataset_dir, "wnids.txt")
        with open(wnids_path, "r", encoding="utf-8") as f:
            self.wnids = [line.strip() for line in f if line.strip()]
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.num_classes = len(self.wnids)

        # 2. 解析 words.txt 获取 WordNet 真实语义
        words_path = os.path.join(self.dataset_dir, "words.txt")
        self.wnid_to_words = {}
        if os.path.exists(words_path):
            with open(words_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        self.wnid_to_words[parts[0]] = parts[1]

        # 3. 预构建文本 Token 字典
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.class_text_tokens = {}
        for idx, wnid in enumerate(self.wnids):
            words_desc = self.wnid_to_words.get(wnid, f"entity {idx}")
            clean_desc = words_desc.replace(",", " ").replace("_", " ").replace("-", " ")
            desc = f"a photo of {clean_desc}"
            tokens = []
            for w in desc.split():
                if w not in self.vocab and len(self.vocab) < self.vocab_size:
                    self.vocab[w] = len(self.vocab)
                tokens.append(self.vocab.get(w, 1))
            tokens = tokens[:seq_len]
            if len(tokens) < seq_len:
                tokens = tokens + [0] * (seq_len - len(tokens))
            self.class_text_tokens[idx] = torch.tensor(tokens, dtype=torch.long)

        # 4. 判断是否存在二进制缓存文件 (若存在直接 mmap/读取，<1 秒完成)
        split_tag = "train" if self.train else "val"
        cache_path = os.path.join(self.dataset_dir, f"{split_tag}_cache.npz")

        if os.path.exists(cache_path):
            print(f"[Dataset] 发现预构建缓存: {cache_path}，正在直接加载...")
            cached = np.load(cache_path)
            self.data = cached["data"]
            self.targets = cached["targets"]
            print(f"[Dataset] TinyImageNet {split_tag} 载入完成 (样本数: {len(self.targets)})。")
        else:
            # 5. 首次加载：多线程快速聚合读取并生成缓存
            file_paths = []
            targets = []

            if self.train:
                train_dir = os.path.join(self.dataset_dir, "train")
                for wnid in self.wnids:
                    cls_idx = self.wnid_to_idx[wnid]
                    cls_img_dir = os.path.join(train_dir, wnid, "images")
                    if os.path.isdir(cls_img_dir):
                        for img_name in sorted(os.listdir(cls_img_dir)):
                            if img_name.lower().endswith((".jpeg", ".jpg", ".png")):
                                file_paths.append(os.path.join(cls_img_dir, img_name))
                                targets.append(cls_idx)
            else:
                val_dir = os.path.join(self.dataset_dir, "val")
                val_anno_path = os.path.join(val_dir, "val_annotations.txt")
                val_img_dir = os.path.join(val_dir, "images")
                with open(val_anno_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            img_name, wnid = parts[0], parts[1]
                            if wnid in self.wnid_to_idx:
                                file_paths.append(os.path.join(val_img_dir, img_name))
                                targets.append(self.wnid_to_idx[wnid])

            self.targets = np.array(targets, dtype=np.int64)
            num_samples = len(file_paths)
            self.data = np.empty((num_samples, 64, 64, 3), dtype=np.uint8)

            print(f"[Dataset] 首次加载：正在使用多线程并发读取 {num_samples} 张图像...")
            max_workers = min(32, (os.cpu_count() or 4) * 2)
            tasks = list(enumerate(file_paths))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for idx, img_arr in executor.map(_read_single_image, tasks, chunksize=100):
                    self.data[idx] = img_arr

            print(f"[Dataset] 正在写入持久化缓存 {cache_path}，后续运行将免去磁盘解码...")
            np.savez(cache_path, data=self.data, targets=self.targets)
            print(f"[Dataset] 缓存保存成功。")

    def _ensure_dataset_exists(self):
        if os.path.exists(os.path.join(self.dataset_dir, "wnids.txt")):
            return

        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, "tiny-imagenet-200.zip")

        if not os.path.exists(zip_path):
            print(f"[Dataset] 未在本地检测到 Tiny-ImageNet-200，正在下载: {self.TINY_URL}")
            urllib.request.urlretrieve(self.TINY_URL, zip_path)
            print("[Dataset] 下载完成。")

        print("[Dataset] 正在解压 Tiny-ImageNet-200 数据集...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        print("[Dataset] 解压就绪。")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        target = int(self.targets[idx])
        text_tokens = self.class_text_tokens[target]
        return (img, text_tokens), target


# ==============================================================================
# 3. 数据划分与加载器获取
# ==============================================================================
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

    ds_name = config.get("dataset", "MultimodalCIFAR100")

    if ds_name == "MultimodalCIFAR100":
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

    elif ds_name == "MultimodalTinyImageNet":
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        seq_len = config.get("text_seq_len", 24)
        vocab_size = config.get("text_vocab_size", 1000)
        train_ds = MultimodalTinyImageNet(root="./data", train=True, transform=train_transform, seq_len=seq_len, vocab_size=vocab_size)
        test_ds = MultimodalTinyImageNet(root="./data", train=False, transform=test_transform, seq_len=seq_len, vocab_size=vocab_size)
        num_classes = 200
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

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