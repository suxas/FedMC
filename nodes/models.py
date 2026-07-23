import torch
import torch.nn as nn


# ============================================================
# SYSU-MM01 Re-ID 模型组件
# ============================================================

class FDP(nn.Module):
    """频域解析器，强迫特征在频域对齐，保证低秩性"""

    def __init__(self, channels):
        super().__init__()
        self.freq_weight = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x):
        fft_x = torch.fft.fft2(x)
        fft_x = fft_x * self.freq_weight
        return torch.fft.ifft2(fft_x).real


class OSBlock(nn.Module):
    """轻量级 Omni-Scale Block (OSNet 核心组件)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, mid, 1, bias=False),
                                   nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, mid, 1, bias=False),
                                   nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
                                   nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, mid, 1, bias=False),
                                   nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
                                   nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
                                   nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False))

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid * 4, mid * 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid * 4, mid * 4, 1, bias=False),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(mid * 4, out_channels, 1, bias=False)
        self.shortcut = nn.Sequential() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1,
                                                                                      bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1, x2, x3, x4 = self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)
        xs = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.fuse(xs * self.gate(xs))
        return self.relu(out + self.shortcut(x))


class ModalEncoder(nn.Module):
    """模态专属编码器：RGB 和 IR 各自独立的输入处理"""

    def __init__(self, dataset, modal_type):
        super().__init__()
        self.modal_type = modal_type
        if dataset in ["MNIST", "CIFAR10"]:
            in_channels = 1 if dataset == "MNIST" else 3
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
            )
            self.out_channels = 32
        else:  # SYSU-MM01
            in_channels = 3 if modal_type == "RGB" else 1
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.out_channels = 64

    def forward(self, x):
        return self.conv(x)


class DualModalFusion(nn.Module):
    """双模态特征融合模块：拼接 RGB 和 IR 特征后降维"""

    def __init__(self, in_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, ir_feat):
        return self.fusion(torch.cat([rgb_feat, ir_feat], dim=1))


class SharedBackbone(nn.Module):
    """全局共享骨干网 (唯一参与 FL 聚合)"""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        if dataset in ["MNIST", "CIFAR10"]:
            self.fdp = FDP(32)
            self.net = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 16, 256),
                nn.ReLU()
            )
            self.out_dim = 256
        else:  # SYSU-MM01
            self.fdp = FDP(64)
            self.net = nn.Sequential(
                OSBlock(64, 128),
                nn.MaxPool2d(2, 2),
                OSBlock(128, 256),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True)
            )
            self.out_dim = 512

    def forward(self, x):
        return self.net(self.fdp(x))


class ClassifierHead(nn.Module):
    """个性化分类头 (留存本地，不参与聚合)"""

    def __init__(self, dataset, num_classes):
        super().__init__()
        if dataset in ["MNIST", "CIFAR10"]:
            self.fc = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        else:  # SYSU-MM01
            self.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.fc(x)


# ============================================================
# MNIST / CIFAR10 专用分类模型（保留用于非联邦场景）
# ============================================================

class SimpleCNN(nn.Module):
    """轻量 CNN，适用于 MNIST"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class VGG11BN(nn.Module):
    """VGG11-BN，适用于 CIFAR10"""

    def __init__(self, num_classes=10):
        super().__init__()
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        layers = []
        in_ch = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
