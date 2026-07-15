import torch


def get_base_config():
    """全局配置：多模态比例、网络层数、异构分布和截断门限等环境参数"""
    return {
        "dataset": "CIFAR10",               # 数据集选择："FashionMNIST", "CIFAR10", "SYSU-MM01"
        "input_size": 32,                   # 输入图像尺寸（FashionMNIST=28, CIFAR10=32, SYSU-MM01=64）
        "global_rounds": 300,               # 通信轮数（增加到300观察长期收敛）
        "num_sbs": 3,                       # 小基站数量
        "clients_per_sbs": 10,              # 每个SBS的客户端数

        "modality_rgb_ratio": 0.5,          # RGB模态比例

        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "local_lr": 0.01,                   # 本地学习率
        "local_epochs": 3,                  # 本地训练轮数（增加本地漂移量）

        # Non-IID参数（α越大越均匀）
        "alpha_inter": 0.3,                 # SBS间异构程度
        "alpha_intra": 0.3,                 # SBS内异构程度

        "straggler_rate": 0.5,              # 掉队概率（提高到70%，让FedAvg参与者锐减）
        "noma_threshold": 0.2,              # NOMA信道解码下限
        "truncation_threshold": 0.5,        # 截断门限（FedAvg/SALF用）

        # SALF参数
        "salf_up_to_layer": 2,              # 掉队者更新的最后层数

        # Proposed算法优化参数
        "global_lr_init": 1.0,              # 全局学习率初始值
        "global_lr_decay": 0.9998,          # 全局学习率衰减因子（每轮，更慢衰减）

        "use_cache": True                   # 是否使用缓存
    }