import torch


def get_base_config():
    """获取基础配置（以 test.py 理想参数为基准）

    SYSU-MM01 跨模态 Re-ID 场景：
    - 模型：DualStreamReID（OSNet + GroupNorm）
    - 优化器：Adam, lr=0.0003
    - 数据划分：簇间 0.5 IID，簇内完全 IID
    - 掉队比例：straggler_rate
    - 聚合方式：OTA（空中计算）
    """
    config = {
        "dataset": "SYSU-MM01",
        "num_sbs": 3,
        "clients_per_sbs": 5,
        "client_fraction": 1.0,

        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # 数据划分：簇间 0.5 IID，簇内完全 IID
        "alpha_inter": 0.5,
        "alpha_intra": 1.0,      # 完全 IID

        # 掉队模拟（时间窗）
        "straggler_rate": 0.3,
        "t_deadline": 99999.0,

        # SALF参数
        "salf_up_to_layer": 2,

        # Proposed参数
        "global_lr_init": 1.0,
        "global_lr_decay": 1,
        "recovery_weight": 0.7,
        "buffer_weight": 0.3,

        "use_cache": False,

        # SYSU-MM01 专用参数
        "input_size": 128,        # 降低分辨率提速
        "batch_size": 64,
        "local_lr": 0.001,        # SGD 学习率
        "reid_lr": 0.001,         # Re-ID 专用学习率（SGD）
        "local_epochs": 1,        # Re-ID 本地训练轮数
        "global_rounds": 1,
        "data_path": "./data/SYSU-MM01",
        "pk_P": 8,                # 每batch人数（增大提升GPU利用率）
        "pk_K": 4,                # 每人图片数（增大提升GPU利用率）
    }

    return config
