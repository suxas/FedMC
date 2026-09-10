import torch

def get_base_config():
    return {
        # 数据集配置: "MultimodalFlowers102" 或 "MultimodalCUB200"
        "dataset": "MultimodalFlowers102",
        "num_classes": 102,               # Flowers102: 102 | CUB200: 200
        "text_seq_len": 24,               # Flowers102: 24  | CUB200: 32
        "text_vocab_size": 2000,          # Flowers102: 2000| CUB200: 2500
        "global_rounds": 500,             # 全局通信轮次
        "num_sbs": 3,                     # 边缘小基站数量
        "clients_per_sbs": 10,            # 每个SBS下属UAV数量 (共30个设备)
        "seed": 42,                       # 仿真主随机种子

        # 硬件与局部训练参数
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 64,
        "local_lr": 0.01,                 # 本地 SGD 学习率
        "weight_decay": 1e-4,             # L2 正则项
        "local_epochs": 4,                # 默认本地迭代轮数
        "momentum": 0.9,
        "num_workers": 0,

        # FedAT (异步分层联邦) 参数 (IEEE/ACM SC 2020)
        "fedat_gamma": 0.5,              # 陈旧度衰减幂指数 alpha(tau) = (tau + 1)^(-gamma)
        "fedat_tier_period": 2,          # 慢梯队跨轮更新周期 (每2轮完成并提交一次)

        # 数据分布与掉队率控制
        "inner_client_iid": 0.5,         # 簇内 0.5 IID + 0.5 Non-IID
        "straggler_ratio": 0.5,          # 掉队设备比例 (50%)

        # Proposed 算法参数
        "compensation_weight": 0.75,     # 历史动量与 LRMC 融合权重
        "buffer_size": 6,                # SBS 历史梯度滑动窗口大小
        "lrmc_rank_ratio_min": 0.80,     # 训练前期能量截断比 (低秩去噪)
        "lrmc_rank_ratio_max": 0.95,     # 训练后期能量截断比 (平滑退火，保留细粒度特征)
        "lrmc_max_iter": 40,             # LRMC 最大迭代次数
        "lrmc_tol": 1e-4,

        # OTA 空中计算信道参数 (显式开关解耦)
        "enable_ota_noise": False,       # 是否注入无线物理信道白噪声
        "ota_snr_db": 20.0,              # 开启加噪时的基准健康信噪比 (20.0 dB)

        # 评估配置
        "eval_batch_size": 128,
        "use_cache": False,
    }