import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    FashionMNIST使用的模型
    """

    def __init__(self, in_channels=1, img_size=28, num_classes=10, kernel=5, ch1=6, ch2=50, **kwargs):
        super().__init__()

        # 卷积块 1 (保持空间尺寸不变: 28x28 -> 28x28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # 28x28 -> 14x14

        # 卷积块 2 (14x14 -> 12x12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 卷积块 3 (12x12 -> 10x10)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)  # 10x10 -> 5x5

        # 动态推导全连接层输入维度 (默认 28x28 输入对应 5 * 5 * 64)
        size = img_size // 2  # pool1: 14
        size = size - 2  # conv2: 12
        size = size - 2  # conv3: 10
        size = size // 2  # pool2: 5
        fc_in_features = size * size * 64

        # 全连接输出层
        self.fc = nn.Linear(fc_in_features, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ---- VGG for CIFAR-10 ----

_vgg_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class VGG11(nn.Module):
    """适用于 CIFAR10 (3x32x32) 的 VGG11 模型"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = self._make_layers(_vgg_cfg)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

    @staticmethod
    def _make_layers(cfg):
        layers, ch_in = [], 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.extend([
                    nn.Conv2d(ch_in, v, 3, padding=1),
                    nn.GroupNorm(32, v),
                    nn.ReLU(inplace=True),
                ])
                ch_in = v
        layers.append(nn.AvgPool2d(1, 1))
        return nn.Sequential(*layers)


class FederatedModel(nn.Module):
    """统一的联邦学习模型接口"""

    def __init__(self, dataset, num_classes):
        super().__init__()
        if dataset == "CIFAR10":
            self.backbone = VGG11(num_classes)
        else:
            self.backbone = SimpleCNN(in_channels=1, img_size=28, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)
