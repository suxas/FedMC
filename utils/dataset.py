import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

def build_vocab_from_texts(captions, vocab_size=2000):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    word_counts = {}
    for cap in captions:
        words = cap.lower().replace(",", " ").replace(".", " ").replace(";", " ").split()
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    for w, _ in sorted_words:
        if len(vocab) >= vocab_size:
            break
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab

def tokenize_caption(text, vocab, seq_len):
    words = text.lower().replace(",", " ").replace(".", " ").replace(";", " ").split()
    tokens = [vocab.get(w, vocab["<UNK>"]) for w in words][:seq_len]
    if len(tokens) < seq_len:
        tokens = tokens + [vocab["<PAD>"]] * (seq_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

def _read_and_resize(args):
    idx, path, size = args
    with Image.open(path) as img:
        img_rgb = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return idx, np.asarray(img_rgb, dtype=np.uint8)

class MultimodalFlowers102(Dataset):
    def __init__(self, root="./data", train=True, transform=None, seq_len=24, vocab_size=2000):
        if root.rstrip("/\\").endswith("flowers102"):
            self.root = os.path.abspath(root)
        else:
            self.root = os.path.abspath(os.path.join(root, "flowers102"))
        self.train = train
        self.transform = transform
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = 102
        split_tag = "train" if train else "test"
        cache_path = os.path.join(self.root, f"flowers102_{split_tag}_cache.npz")
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            self.data = cached["data"]
            self.targets = cached["targets"]
            self.tokens = torch.from_numpy(cached["tokens"])
        else:
            self._build_from_local(split_tag, cache_path)

    def _build_from_local(self, split_tag, cache_path):
        labels_mat = os.path.join(self.root, "imagelabels.mat")
        setid_mat = os.path.join(self.root, "setid.mat")
        jpg_dir = os.path.join(self.root, "jpg")

        all_labels = scipy.io.loadmat(labels_mat)["labels"][0] - 1
        setid = scipy.io.loadmat(setid_mat)

        if self.train:
            split_indices = np.concatenate([setid["trnid"][0], setid["valid"][0]]) - 1
        else:
            split_indices = setid["tstid"][0] - 1

        img_paths = [os.path.join(jpg_dir, f"image_{idx + 1:05d}.jpg") for idx in split_indices]
        self.targets = np.array([all_labels[idx] for idx in split_indices], dtype=np.int64)
        num_samples = len(img_paths)

        tasks = [(i, p, 64) for i, p in enumerate(img_paths)]
        self.data = np.empty((num_samples, 64, 64, 3), dtype=np.uint8)
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, arr in executor.map(_read_and_resize, tasks, chunksize=100):
                self.data[idx] = arr

        captions = self._generate_sample_descriptions(self.data)
        self.vocab = build_vocab_from_texts(captions, vocab_size=self.vocab_size)
        token_list = [tokenize_caption(c, self.vocab, self.seq_len) for c in captions]
        self.tokens = torch.stack(token_list)
        np.savez(cache_path, data=self.data, targets=self.targets, tokens=self.tokens.numpy())

    def _generate_sample_descriptions(self, images_np):
        imgs = images_np.astype(np.float32)
        r = imgs[:, :, :, 0].mean(axis=(1, 2))
        g = imgs[:, :, :, 1].mean(axis=(1, 2))
        b = imgs[:, :, :, 2].mean(axis=(1, 2))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        captions = []
        for i in range(len(imgs)):
            r_val, g_val, b_val, l_val = r[i], g[i], b[i], lum[i]
            if r_val > g_val + 15 and r_val > b_val + 15:
                hue = "vivid scarlet crimson petals"
            elif r_val > 140 and g_val > 140 and b_val < 100:
                hue = "bright golden yellow blossoms"
            elif b_val > r_val + 10 and b_val > g_val:
                hue = "delicate violet lavender petals"
            elif r_val > 130 and b_val > 130 and g_val < 110:
                hue = "radiant magenta purple corolla"
            elif l_val > 165:
                hue = "pure creamy white floral petals"
            else:
                hue = "rich polychromatic multi-hued petals"
            light = "sunlit outdoor clarity" if l_val > 130 else "soft diffused foliage shade"
            captions.append(
                f"botanical floral specimen displaying {hue} illuminated by {light} with centered flower head"
            )
        return captions

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.tokens[idx]), int(self.targets[idx])

class MultimodalCUB200(Dataset):
    def __init__(self, root="./data", train=True, transform=None, seq_len=32, vocab_size=2500):
        if root.rstrip("/\\").endswith("cub200"):
            self.root = os.path.abspath(root)
        else:
            self.root = os.path.abspath(os.path.join(root, "cub200"))
        self.train = train
        self.transform = transform
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = 200
        split_tag = "train" if train else "test"
        cache_path = os.path.join(self.root, f"cub200_{split_tag}_cache.npz")
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            self.data = cached["data"]
            self.targets = cached["targets"]
            self.tokens = torch.from_numpy(cached["tokens"])
        else:
            self._build_and_cache(split_tag, cache_path)

    def _build_and_cache(self, split_tag, cache_path):
        base_dir = os.path.join(self.root, "CUB_200_2011")
        images_txt = os.path.join(base_dir, "images.txt")
        labels_txt = os.path.join(base_dir, "image_class_labels.txt")
        split_txt = os.path.join(base_dir, "train_test_split.txt")

        id_to_path, id_to_label, is_train_id = {}, {}, {}
        with open(images_txt, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 2:
                    id_to_path[int(p[0])] = os.path.join(base_dir, "images", p[1])
        with open(labels_txt, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 2:
                    id_to_label[int(p[0])] = int(p[1]) - 1
        with open(split_txt, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 2:
                    is_train_id[int(p[0])] = (int(p[1]) == 1)

        selected_ids = [img_id for img_id, is_tr in is_train_id.items() if is_tr == self.train]
        selected_ids.sort()
        img_paths = [id_to_path[i] for i in selected_ids]
        self.targets = np.array([id_to_label[i] for i in selected_ids], dtype=np.int64)

        attr_names_file = os.path.join(base_dir, "attributes", "attributes.txt")
        attr_labels_file = os.path.join(base_dir, "attributes", "image_attribute_labels.txt")
        attr_id_to_name = {}
        if os.path.exists(attr_names_file):
            with open(attr_names_file, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 2:
                        attr_id_to_name[int(p[0])] = p[1].replace("has_", "").replace("::", " ").replace("_", " ")

        image_attrs = {img_id: [] for img_id in selected_ids}
        if os.path.exists(attr_labels_file):
            selected_set = set(selected_ids)
            with open(attr_labels_file, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 4:
                        img_id = int(p[0])
                        if img_id in selected_set and int(p[2]) == 1:
                            name = attr_id_to_name.get(int(p[1]))
                            if name:
                                image_attrs[img_id].append(name)

        captions = []
        for img_id in selected_ids:
            obs = image_attrs.get(img_id, [])
            desc = "a fine-grained avian species characterized by " + ", and ".join(obs[:6]) if len(obs) >= 4 else \
                   "a wild bird perched outdoors exhibiting distinct plumage markings and beak structure"
            captions.append(desc)

        num_samples = len(img_paths)
        tasks = [(i, p, 64) for i, p in enumerate(img_paths)]
        self.data = np.empty((num_samples, 64, 64, 3), dtype=np.uint8)
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, arr in executor.map(_read_and_resize, tasks, chunksize=100):
                self.data[idx] = arr

        self.vocab = build_vocab_from_texts(captions, vocab_size=self.vocab_size)
        token_list = [tokenize_caption(c, self.vocab, self.seq_len) for c in captions]
        self.tokens = torch.stack(token_list)
        np.savez(cache_path, data=self.data, targets=self.targets, tokens=self.tokens.numpy())

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.tokens[idx]), int(self.targets[idx])

def split_data_dmfl(targets, num_sbs, clients_per_sbs, inner_client_iid, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    num_users = num_sbs * clients_per_sbs
    u_per = clients_per_sbs
    labels = np.asarray(targets)
    idx = np.arange(len(labels))
    sorted_idx = idx[labels.argsort()]

    # 1. 簇外严格 Non-IID: 每个 SBS 分配完全独立的类别分片
    shards = np.array_split(sorted_idx, num_sbs)
    user_map = {i: np.array([], dtype='int64') for i in range(num_users)}

    for eid in range(num_sbs):
        shard = shards[eid]
        total = len(shard)
        # 2. 簇内 0.5 IID + 0.5 Non-IID
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ds_name = config.get("dataset", "MultimodalFlowers102")
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

    if ds_name == "MultimodalFlowers102":
        seq_len = config.get("text_seq_len", 24)
        vocab_size = config.get("text_vocab_size", 2000)
        train_ds = MultimodalFlowers102(root="./data", train=True, transform=train_transform, seq_len=seq_len, vocab_size=vocab_size)
        test_ds = MultimodalFlowers102(root="./data", train=False, transform=test_transform, seq_len=seq_len, vocab_size=vocab_size)
        num_classes = 102
    elif ds_name == "MultimodalCUB200":
        seq_len = config.get("text_seq_len", 32)
        vocab_size = config.get("text_vocab_size", 2500)
        train_ds = MultimodalCUB200(root="./data", train=True, transform=train_transform, seq_len=seq_len, vocab_size=vocab_size)
        test_ds = MultimodalCUB200(root="./data", train=False, transform=test_transform, seq_len=seq_len, vocab_size=vocab_size)
        num_classes = 200
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    targets = np.asarray(train_ds.targets)
    inner_client_iid = config.get("inner_client_iid", 0.5)
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