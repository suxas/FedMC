import torch


def get_base_config():
    """全局配置文件，定义了多模态比例、网络层数、异构分布和截断门限等环境参数"""
    return {
        "dataset": "FashionMNIST",  # 可选 "CIFAR10" 或 "FashionMNIST"
        "global_rounds": 30,  # 全局通信轮数
        "num_sbs": 3,  # 小基站 (SBS) / 簇头数量
        "clients_per_sbs": 10,  # 每个簇内的边缘节点数量

        # 多模态传感器载荷分布
        "modality_rgb_ratio": 0.5,  # 50% 节点搭载 RGB 镜头，剩余 50% 搭载多光谱 NIR 镜头

        # 训练参数
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "local_lr": 0.01,
        "local_epochs": 1,

        # 狄利克雷 Non-IID 异构性 (越小越 Non-IID)
        "alpha_inter": 0.5,
        "alpha_intra": 0.5,

        # 物理信道与截断约束
        "straggler_rate": 0.5,  # 掉队设备基础概率
        "noma_threshold": 0.2,  # NOMA 信道解码下限
        "truncation_threshold": 0.6,  # 严格的时间/功率截断门限 (传统 OMA 使用)

        # 实验辅助控制
        "use_cache": True  # 是否使用历史缓存结果避免重复运行
    }