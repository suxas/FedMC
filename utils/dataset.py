import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# 21 个地貌类别的直接泄漏实体词与近义词库
LEAKY_KEYWORDS = {
    # agricultural
    "agricultural", "agriculture", "farmland", "cropland", "farm", "crop", "crops", "plantation",
    # airplane
    "airplane", "airplanes", "plane", "planes", "aircraft", "jet", "jets", "airliner",
    # baseballdiamond
    "baseball", "diamond", "ballfield",
    # beach
    "beach", "beaches", "seashore", "coastline", "shoreline", "sand", "sandy",
    # buildings
    "building", "buildings", "skyscraper", "skyscrapers", "house", "houses", "edifice",
    # chaparral
    "chaparral", "shrub", "shrubs", "shrubbery", "scrub", "scrubland", "brush",
    # residential (dense, medium, sparse, mobile)
    "residential", "neighborhood", "homes", "home", "mobilehome", "trailer", "trailers",
    "dense", "medium", "sparse",
    # forest
    "forest", "forests", "woodland", "woods", "tree", "trees", "wooded",
    # freeway / overpass / intersection
    "freeway", "freeways", "highway", "highways", "expressway", "motorway", "interchange", "intersection", "crossroad", "junction",
    # golfcourse
    "golf", "golfcourse", "fairway", "fairways",
    # harbor
    "harbor", "harbors", "port", "ports", "dock", "docks", "pier", "piers", "wharf", "boat", "boats", "ship", "ships",
    # overpass
    "overpass", "flyover", "bridge",
    # parkinglot
    "parking", "parkinglot", "cars", "car", "vehicles", "vehicle", "parked",
    # river
    "river", "rivers", "stream", "streams", "waterway",
    # runway
    "runway", "runways", "airstrip", "taxiway", "airfield", "airport",
    # storagetanks
    "storage", "tank", "tanks", "silo", "silos", "cistern",
    # tenniscourt
    "tennis", "court", "courts"
}

def mask_caption(text):
    """
    方案一核心：清洗文本并对泄露类别标签的核心实体词进行 <MASK> 掩码替换
    """
    words = text.replace(",", " , ").replace(".", " . ").replace(";", " ; ").replace('"', " ").replace("'", " ").split()
    masked_words = []
    for w in words:
        clean_w = w.lower().strip(".,;:\"'!?()[]{}")
        if clean_w in LEAKY_KEYWORDS:
            masked_words.append("<MASK>")
        else:
            masked_words.append(w)
    return " ".join(masked_words)

def build_vocab_from_texts(captions, vocab_size=2500):
    vocab = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}
    word_counts = {}
    for cap in captions:
        words = cap.lower().replace(",", " ").replace(".", " ").replace(";", " ").replace('"', " ").replace("'", " ").split()
        for w in words:
            if w in vocab:
                continue
            word_counts[w] = word_counts.get(w, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    for w, _ in sorted_words:
        if len(vocab) >= vocab_size:
            break
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab

def tokenize_caption(text, vocab, seq_len):
    words = text.lower().replace(",", " ").replace(".", " ").replace(";", " ").replace('"', " ").replace("'", " ").split()
    tokens = [vocab.get(w, vocab["<UNK>"]) for w in words][:seq_len]
    if len(tokens) < seq_len:
        tokens = tokens + [vocab["<PAD>"]] * (seq_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

class UAVAERDataset(Dataset):
    """
    无人机应急侦察航拍数据集 (平铺结构: imgs/1.tif ~ 2100.tif，集成实体掩码保护)
    """
    CLASSES = [
        "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
        "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
        "harbor", "intersection", "mediumresidential", "mobilehomepark",
        "overpass", "parkinglot", "river", "runway", "sparseresidential",
        "storagetanks", "tenniscourt"
    ]

    def __init__(self, root="./data/uav_aer", train=True, transform=None, seq_len=24, vocab_size=2500):
        self.root = os.path.abspath(root)
        self.train = train
        self.transform = transform
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = len(self.CLASSES)

        split_tag = "train" if train else "test"
        # 更新缓存名，避免与旧版本未掩码的缓存产生冲突
        self.cache_path = os.path.join(self.root, f"uav_aer_flat_masked_{split_tag}_cache.npz")

        if os.path.exists(self.cache_path):
            cached = np.load(self.cache_path)
            self.images = cached["images"]
            self.targets = cached["targets"]
            self.tokens = torch.from_numpy(cached["tokens"])
        else:
            self._prepare_dataset(split_tag)

    def _locate_imgs_dir(self):
        candidates = ["imgs", "Images", "images"]
        for cand in candidates:
            p = os.path.join(self.root, cand)
            if os.path.exists(p) and os.path.isdir(p):
                return p
        raise FileNotFoundError(f"[Dataset 错误] 未在 {self.root} 下找到 imgs 图像目录！")

    def _prepare_dataset(self, split_tag):
        imgs_dir = self._locate_imgs_dir()
        captions_path = os.path.join(self.root, "dataset.json")

        if not os.path.exists(captions_path):
            raise FileNotFoundError(f"[Dataset 错误] 未找到标注文件: {captions_path}")

        print(f"[Dataset] 正在解析标注文件并应用关键词掩码: {captions_path}...")
        with open(captions_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        caption_map = {}
        all_masked_captions = []

        images_list = meta.get("images", [])
        for item in images_list:
            fname = os.path.basename(item.get("filename", "")).lower()
            sents = []
            for s in item.get("sentences", []):
                raw = s.get("raw", "").strip()
                if raw:
                    # 应用实体关键词掩码处理，切断文字捷径
                    masked_sent = mask_caption(raw)
                    sents.append(masked_sent)
                    all_masked_captions.append(masked_sent)
            if fname and sents:
                caption_map[fname] = sents

        print(f"[Dataset] 标注映射建立完毕 (已注入 <MASK> 保护)，匹配图像条目: {len(caption_map)}")

        vocab = build_vocab_from_texts(all_masked_captions, vocab_size=self.vocab_size)

        all_images = []
        all_captions = []
        all_labels = []

        for class_idx in range(len(self.CLASSES)):
            start_num = class_idx * 100 + 1
            if self.train:
                num_range = range(start_num, start_num + 80)
            else:
                num_range = range(start_num + 80, start_num + 100)

            for img_num in num_range:
                f_name = f"{img_num}.tif"
                img_path = os.path.join(imgs_dir, f_name)
                if not os.path.exists(img_path):
                    alt_path = os.path.join(imgs_dir, f"{img_num}.TIF")
                    if os.path.exists(alt_path):
                        img_path = alt_path
                    else:
                        raise FileNotFoundError(f"[Dataset 错误] 缺失真实图像文件: {img_path}")

                sents = caption_map.get(f_name.lower())
                if not sents:
                    raise RuntimeError(f"[Dataset 错误] 图像 {f_name} 在 dataset.json 中缺失标注文本！")

                try:
                    with Image.open(img_path) as img:
                        img_resized = img.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
                        all_images.append(np.asarray(img_resized, dtype=np.uint8))
                        all_labels.append(class_idx)
                        all_captions.append(sents[0])
                except Exception as e:
                    raise RuntimeError(f"[Dataset 错误] 图像文件损坏无法读取: {img_path}, 详情: {e}")

        self.images = np.array(all_images, dtype=np.uint8)
        self.targets = np.array(all_labels, dtype=np.int64)
        token_list = [tokenize_caption(c, vocab, self.seq_len) for c in all_captions]
        self.tokens = torch.stack(token_list)

        np.savez(self.cache_path, images=self.images, targets=self.targets, tokens=self.tokens.numpy())
        print(f"[Dataset] 掩码缓存已保存: {self.cache_path} ({split_tag} 样本数: {len(self.targets)}, 类别数: {self.num_classes})")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.tokens[idx]), int(self.targets[idx])

def split_data_dmfl(targets, num_sbs, clients_per_sbs, inner_client_iid, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    num_users = num_sbs * clients_per_sbs
    u_per = clients_per_sbs
    labels = np.asarray(targets)
    idx = np.arange(len(labels))
    sorted_idx = idx[labels.argsort()]

    # 簇间严格 Non-IID
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

    root_dir = config.get("data_dir", "./data/uav_aer")
    train_ds = UAVAERDataset(
        root=root_dir, train=True, transform=train_transform,
        seq_len=config.get("text_seq_len", 24),
        vocab_size=config.get("text_vocab_size", 2500)
    )
    test_ds = UAVAERDataset(
        root=root_dir, train=False, transform=test_transform,
        seq_len=config.get("text_seq_len", 24),
        vocab_size=config.get("text_vocab_size", 2500)
    )
    num_classes = train_ds.num_classes

    targets = np.asarray(train_ds.targets)
    inner_client_iid = config.get("inner_client_iid", 0.2)
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
        client_gen = torch.Generator()
        client_gen.manual_seed(seed + client_id * 17)
        train_loader = DataLoader(
            train_subset, batch_size=config["batch_size"],
            shuffle=True, generator=client_gen, **loader_kwargs
        )
        train_loaders.append(train_loader)

    test_loader = DataLoader(
        test_ds, batch_size=config.get("eval_batch_size", 64),
        shuffle=False, **loader_kwargs
    )
    return train_loaders, test_loader, num_classes