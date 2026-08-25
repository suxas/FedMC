import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """适用于 FashionMNIST 的 CNN 模型，采用 GroupNorm 适配联邦非 IID 场景"""

    def __init__(self, in_channels=1, img_size=28, num_classes=10):
        super().__init__()
        # 卷积块 1 (28x28 -> 14x14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 卷积块 2 (14x14 -> 12x12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )

        # 卷积块 3 (12x12 -> 5x5)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 全连接分类头
        self.fc = nn.Linear(5 * 5 * 64, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


_vgg_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class VGG11(nn.Module):
    """适用于 CIFAR10 的 VGG11 模型"""

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
                    nn.GroupNorm(16, v),
                    nn.ReLU(inplace=True),
                ])
                ch_in = v
        layers.append(nn.AvgPool2d(1, 1))
        return nn.Sequential(*layers)


class FederatedModel(nn.Module):
    """统一模型入口"""

    def __init__(self, dataset, num_classes):
        super().__init__()
        if dataset == "CIFAR10":
            self.backbone = VGG11(num_classes)
        else:
            self.backbone = SimpleCNN(in_channels=1, img_size=28, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)