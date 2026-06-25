import torch


def get_base_config():
    return {
        "dataset": "FashionMNIST",  # 可选: "FashionMNIST", "CIFAR10"
        "global_rounds": 10,  # 为快速演示设为10，写论文建议设为 50-100
        "num_sbs": 3,  # 边缘服务器数量
        "clients_per_sbs": 10,  # 每个 SBS 挂载的设备数

        # --- 真实深度学习配置 ---
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "local_lr": 0.01,  # 客户端 SGD 学习率
        "local_epochs": 1,  # 客户端本地训练轮次

        # --- 两级 Non-IID 数据异构性配置 ---
        # 狄利克雷分布参数，越小 (如 0.1) 越 Non-IID；越大 (如 10.0) 越趋近 IID
        "alpha_inter": 0.5,  # 簇间 (Edge-Cloud) 异构程度
        "alpha_intra": 0.5,  # 簇内 (User-Edge) 异构程度

        # --- 系统异构性与掉队模拟配置 ---
        "straggler_rate": 0.4,  # 40% 的设备无法按时完算/传完

        # --- 物理通信层约束 ---
        "noma_threshold": 0.2,  # NOMA允许叠加传输的最低信道门槛
        "oma_threshold": 0.6  # 传统 OMA 严格截断信道门槛
    }