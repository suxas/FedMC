import torch


def get_base_config():
    return {
        # 数据与实验规模
        "dataset": "FashionMNIST",      # FashionMNIST / CIFAR10
        "global_rounds": 300,           # 全局通信轮数
        "num_sbs": 3,                   # 边缘基站数量
        "clients_per_sbs": 10,           # 每个基站的客户端数量
        "seed": 42,                     # 全局随机种子

        # 本地训练参数
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 64,               # 参考DMFL
        "local_lr": 0.001,              # 参考DMFL，降低学习率
        "local_epochs": 3,
        "momentum": 0.5,                # 参考DMFL，SGD动量
        "num_workers": 0,

        # 数据划分异构性（
        "inner_client_iid": 0.1,        # edge内IID数据比例 (0-1)，0.1表示10%IID+90%Non-IID

        # 掉队模拟：统一随机掩码
        "straggler_ratio": 0.5,
        "compute_time_mean": 0.5,
        "compute_time_std": 0.2,
        "transmit_time_mean": 0.3,
        "transmit_time_std": 0.1,

        # Proposed 算法：LRMC 与缓冲池
        "compensation_weight": 0.9,     # LRMC 补偿方向权重
        "buffer_size": 5,               # 历史梯度缓冲长度
        "lrmc_rank_ratio": 0.8,         # 低秩补全能量保留比例
        "lrmc_max_iter": 30,
        "lrmc_tol": 1e-4,

        # OTA 空中接口
        "ota_snr_db": 0.0,             # 0 表示不加噪

        # SALF 掉队客户端上传语义模块数
        "salf_up_to_layer": 2,

        # 评估参数
        "eval_batch_size": 128,
    }
