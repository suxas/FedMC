import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """适用于 FashionMNIST (1x28x28) 的轻量 CNN"""

    def __init__(self, in_channels=1, img_size=28, num_classes=10, kernel=5, ch1=6, ch2=50):
        super().__init__()
        self.ch1 = ch1
        size = img_size

        self.conv1 = nn.Conv2d(in_channels, ch1, kernel)
        size = (size - kernel + 1) // 2

        self.conv2 = nn.Conv2d(ch1, ch1, kernel)
        size = int((size - kernel + 1) // 2)

        self.fc1 = nn.Linear(ch1 * size * size, ch2)
        self.fc2 = nn.Linear(ch2, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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
