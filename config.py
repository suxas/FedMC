import torch

def get_base_config():
    return {
        # 数据集与任务配置
        "dataset": "UAV-AER",
        "num_classes": 21,
        "text_seq_len": 24,
        "text_vocab_size": 2500,
        "global_rounds": 100,
        "num_sbs": 3,
        "clients_per_sbs": 10,
        "seed": 42,

        # 数据集获取与校验
        "auto_download": True,
        "image_url": "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip",
        "captions_url": "https://raw.githubusercontent.com/BigData-KSU/UCMerced-LandUse-Captions/master/dataset.json",

        # 硬件与本地训练超参数
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "local_lr": 0.01,
        "weight_decay": 1e-4,
        "local_epochs": 4,
        "momentum": 0.9,
        "num_workers": 0,

        # FedAT 异步分层参数
        "fedat_gamma": 0.5,
        "fedat_tier_period": 4,

        # FedGC 前沿动量梯度补偿基准参数
        "fedgc_momentum": 0.8,

        # 数据拓扑与掉队设置
        "inner_client_iid": 0.2,
        "straggler_ratio": 0.5,

        # Proposed 核心算法参数
        "compensation_weight": 0.75,
        "buffer_size": 6,
        "lrmc_rank_ratio_min": 0.80,
        "lrmc_rank_ratio_max": 0.95,
        "lrmc_max_iter": 30,
        "lrmc_tol": 1e-4,

        # 空中计算通信参数
        "enable_ota_noise": False,
        "ota_snr_db": 20.0,

        # 李雅普诺夫在线时延优化控制参数
        "lyapunov_compression_ratio": 0.78,
        "lyapunov_v": 90.0,

        # 模型验证评估配置
        "eval_batch_size": 64,
        "use_cache": False,
    }