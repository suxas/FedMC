import torch
import torch.nn as nn


# ============================================================
# 模型配置：每个数据集独立定义，复杂度递增
# ============================================================

MODEL_CONFIGS = {
    "MNIST": {
        "conv_channels": [16, 32],
        "use_bn": False,
        "pool_size": (4, 4),           # 32×32 → pool后 4×4
        "classifier_hidden": 64,
        "classifier_dropout": 0.0,
    },
    "CIFAR10": {
        "conv_channels": [32, 64, 128],
        "use_bn": True,
        "pool_size": (4, 4),           # 32×32 → pool后 4×4
        "classifier_hidden": 256,
        "classifier_dropout": 0.3,
    },
    "CIFAR100": {
        "conv_channels": [64, 128, 256, 256],
        "use_bn": True,
        "pool_size": (4, 4),           # 32×32 → pool后 4×4
        "classifier_hidden": 512,
        "classifier_dropout": 0.4,
    },
}


# ============================================================
# 通用模块
# ============================================================

class FeatureExtractor(nn.Module):
    """特征提取器：根据数据集配置动态构建卷积层"""

    def __init__(self, in_channels, model_cfg):
        super().__init__()
        channels = model_cfg["conv_channels"]
        use_bn = model_cfg["use_bn"]
        pool_size = model_cfg["pool_size"]

        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.append(nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU())
            prev_ch = ch
        layers.append(nn.AdaptiveAvgPool2d(pool_size))
        layers.append(nn.Flatten())

        self.conv = nn.Sequential(*layers)
        self._feature_dim = channels[-1] * pool_size[0] * pool_size[1]

    def forward(self, x):
        return self.conv(x)

    def get_feature_dim(self):
        return self._feature_dim


class ClassifierHead(nn.Module):
    """分类头：根据数据集配置动态构建全连接层"""

    def __init__(self, in_features, num_classes, model_cfg):
        super().__init__()
        hidden = model_cfg["classifier_hidden"]
        dropout = model_cfg["classifier_dropout"]

        layers = [nn.Linear(in_features, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, num_classes))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


# ============================================================
# 工厂函数
# ============================================================

def get_model_config(dataset_name):
    """获取数据集对应的模型配置"""
    if dataset_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[dataset_name]
    raise ValueError(f"未知数据集: {dataset_name}, 可选: {list(MODEL_CONFIGS.keys())}")


def create_feature_extractor(dataset_name, in_channels=3):
    """根据数据集名称创建特征提取器"""
    cfg = get_model_config(dataset_name)
    return FeatureExtractor(in_channels, cfg)


def create_classifier(dataset_name, feature_dim, num_classes):
    """根据数据集名称创建分类头"""
    cfg = get_model_config(dataset_name)
    return ClassifierHead(feature_dim, num_classes, cfg)
