def get_base_config():
    """所有算法共享的物理底座参数与超参数"""
    return {
        "dataset": "FashionMNIST",
        "global_rounds": 50,  # MBS 全局聚合的轮次
        "num_sbs": 3,  # 边缘服务器 (SBS) 数量
        "clients_per_sbs": 10,  # 每个 SBS 挂载的设备数
        "grad_dim": 500,  # 特征层梯度展平后的维度

        # clients 配置
        "local_lr": 0.01,  # 客户端本地SGD学习率
        "local_epochs": 5,  # 客户端在单次通信前，在本地进行多少轮完整迭代

        "straggler_rate": 0.4,  # 整体设备的基准掉队率 (如 0.4 代表每轮有 40% 的设备无法在 T_win 内完算/传完)

        "noma_threshold": 0.2,  # NOMA允许叠加传输的最低信道门槛
        "oma_threshold": 0.6  # 传统 OMA 能够按时传完的严格信道门槛
    }