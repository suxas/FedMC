import torch


def get_base_config():
    return {
        "dataset": "FashionMNIST",  # "FashionMNIST" or "CIFAR10"
        "global_rounds": 30,
        "num_sbs": 3,
        "clients_per_sbs": 10,

        # 训练参数
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "local_lr": 0.01,
        "local_epochs": 1,

        # 两级 Non-IID 异构性（越小越 Non-IID）
        "alpha_inter": 0.5,
        "alpha_intra": 0.5,

        # 系统异构性
        "straggler_rate": 0.4,

        # 通信约束
        "noma_threshold": 0.2,  # NOMA 叠加传输最低信道门限
        "oma_threshold": 0.6,   # OMA 严格截断信道门限

        # SALF 掉队者只更新最后 N 层（从末尾算起）
        "salf_up_to_layer": 2
    }
