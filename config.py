import torch


def get_base_config():
    return {
        # 数据与实验规模
        "dataset": "MultimodalCIFAR100",  # 多模态 CIFAR-100 图文基准
        "num_classes": 100,               # CIFAR-100 包含 100 个类别
        "text_seq_len": 16,               # 文本 Prompt 序列长度
        "text_vocab_size": 500,           # 词表容量
        "global_rounds": 500,            # 全局通信轮数
        "num_sbs": 3,                     # 边缘基站数量
        "clients_per_sbs": 10,            # 每个基站的客户端数量
        "seed": 42,                       # 全局随机种子

        # 本地训练参数 (适配 13765HX + RTX 5060 Laptop)
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 64,
        "local_lr": 0.01,                 # 固定学习率 0.01
        "weight_decay": 1e-4,             # L2 正则化
        "local_epochs": 3,
        "momentum": 0.9,
        "num_workers": 0,

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

        # SALF 掉队客户端上传语义模块数
        "salf_up_to_layer": 2,

        # 评估参数
        "eval_batch_size": 128,
        "use_cache": False,
    }