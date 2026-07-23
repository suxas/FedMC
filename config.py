import torch


def get_base_config():
    return {
        # --- 数据集设置 ---
        "dataset": "SYSU-MM01",  # 可选: "MNIST", "CIFAR10", "SYSU-MM01"
        "global_rounds":  100,
        "num_sbs": 3,
        "clients_per_sbs": 10,

        # 模态异构设置（簇内 UAV 模态分配比例）
        "modal_split_ratio": {"RGB": 0.3, "IR": 0.3, "Both": 0.4},

        # --- 深度学习设置 ---
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "local_lr": 0.001,
        "local_epochs": 1,
        "num_workers": 0,          # DataLoader 工作进程数（Windows建议0）

        # --- 数据异构性 ---
        "alpha_inter": 0.5,   # 簇间 IID 程度（0.5 = 半IID）
        "alpha_intra": 1.0,   # 簇内数据重叠程度

        # --- 掉队模拟（时间窗口） ---
        "straggler_ratio": 0.3,    # 掉队比例（默认30%）
        "time_window": 1.0,        # 时间窗口阈值
        "compute_time_mean": 0.5,  # 计算时间均值
        "compute_time_std": 0.2,   # 计算时间标准差
        "transmit_time_mean": 0.3, # 传输时延均值
        "transmit_time_std": 0.1,  # 传输时延标准差

        # --- SBS 缓冲池 ---
        "compensation_weight": 0.7,  # LRMC补偿梯度权重
        "history_weight": 0.3,       # 历史梯度权重
        "buffer_size": 5,            # 历史梯度缓冲池大小

        # --- LRMC 梯度补全 ---
        "lrmc_rank_ratio": 0.5,   # SVD 截断奇异值比例
        "lrmc_max_iter": 50,      # LRMC 最大迭代次数
        "lrmc_tol": 1e-4,         # LRMC 收敛阈值

        # --- OTA 通信 ---
        "ota_snr_db": 30.0,       # OTA 空中计算的信噪比 (dB)
        "ota_drop_rate": 0.0,     # OTA 聚合失败概率

        # --- SALF ---
        "salf_up_to_layer": 2,    # SALF 掉队者上传层数

        # --- 评估 ---
        "eval_every": 5,           # 每 N 轮评估一次（提高训练速度）
    }
