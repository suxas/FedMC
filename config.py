import torch


def get_base_config():
    return {
        # --- 数据集设置 ---
        "dataset": "SYSU-MM01",  # 可选: "MNIST", "CIFAR10", "SYSU-MM01"
        "global_rounds": 100,
        "num_sbs": 3,
        "clients_per_sbs": 10,

        # 模态异构设置
        "modal_ratio": {"RGB": 0.5, "IR": 0.5},

        # --- 深度学习设置 ---
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "local_lr": 0.001,       # Re-ID 需要较小学习率
        "local_epochs": 1,

        # --- 数据异构性 ---
        "alpha_inter": 0.8,
        "alpha_intra": 1,

        # --- OTA 通信与算法机制 ---
        "ota_snr_db": 30.0,  # OTA 空中计算的信噪比 (dB)
        "ota_drop_rate": 0,  # OTA 聚合完全失败的概率（模拟严重衰落）
        "time_window_size": 5,  # main.py 中时间窗缓存池大小
        "salf_up_to_layer": 2,  # main_salf.py 掉队者上传层数
        "eval_every": 5,  # 每 N 轮评估一次
    }