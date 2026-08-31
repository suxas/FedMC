import torch


def get_base_config():
    return {
        # 数据集选择: "MultimodalCIFAR100" (100类, 32x32) 或 "MultimodalTinyImageNet" (200类, 64x64)
        "dataset": "MultimodalTinyImageNet",
        "num_classes": 200,               # CIFAR100: 100 | TinyImageNet: 200
        "text_seq_len": 24,               # CIFAR100: 16  | TinyImageNet: 24
        "text_vocab_size": 1000,          # CIFAR100: 500 | TinyImageNet: 1000
        "global_rounds": 300,             # 全局通信轮数
        "num_sbs": 3,                     # 边缘基站数量
        "clients_per_sbs": 10,            # 每个基站的客户端数量
        "seed": 42,                       # 全局随机种子

        # 本地训练参数
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 128,                 # CIFAR100: 64 | TinyImageNet: 128
        "local_lr": 0.01,                 # 学习率
        "weight_decay": 1e-4,             # L2 正则化
        "local_epochs": 3,
        "momentum": 0.9,
        "num_workers": 0,

        # FedProx 近端正则系数
        "fedprox_mu": 0.01,

        # 数据划分异构性 (簇间 Non-IID, 簇内近似 IID)
        "inner_client_iid": 0.1,          # 10% 全局 IID 混合池 + 90% 簇内分片

        # 掉队模拟
        "straggler_ratio": 0.5,

        # Proposed 算法：LRMC 与缓冲池
        "compensation_weight": 0.8,       # 基础补偿权重
        "buffer_size": 5,                 # 历史梯度缓冲深度
        "lrmc_rank_ratio": 0.8,           # 能量保留比例
        "lrmc_max_iter": 30,
        "lrmc_tol": 1e-4,

        # OTA 空中接口 (0 表示无加噪)
        "ota_snr_db": 0.0,

        # 评估参数
        "eval_batch_size": 128,
        "use_cache": False,
    }