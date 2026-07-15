import torch


def get_base_config():
    """全局配置：无人机联邦学习场景参数"""
    return {
        "dataset": "MNIST",                  # 数据集选择："MNIST", "CIFAR10", "CIFAR100"
        "input_size": 32,                    # 输入图像尺寸（统一32×32）
        "global_rounds": 50,                # 全局通信轮数
        "num_sbs": 3,                        # 小基站（SBS）数量
        "clients_per_sbs": 10,               # 每个SBS管理的无人机（UAV）数量
        "client_fraction": 1.0,              # 每轮随机选择的客户端比例（FedAvg原版参数C）

        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "local_lr": 0.01,                    # 本地学习率
        "local_epochs": 3,                   # 本地训练轮数

        # 数据分布参数
        "alpha_intra": 0.1,                  # 簇内数据重叠比例（0=完全不重叠，1=完全IID，0.1=10%重叠）

        # 掉队模拟参数
        "straggler_rate": 0.5,               # 基础掉队概率
        "channel_threshold": 0.3,            # 信道质量门限（低于此值视为无法通信）
        "truncation_threshold": 0.5,         # 截断门限（FedAvg/SALF用）

        # SALF参数
        "salf_up_to_layer": 2,               # 掉队者更新的最后层数

        # Proposed算法优化参数
        "global_lr_init": 1.0,               # 全局学习率初始值
        "global_lr_decay": 1,           # 全局学习率衰减因子

        "use_cache": True                    # 是否使用缓存
    }
