import torch
import torch.nn as nn


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
        # 多尺度特征流 (Depthwise Separable)
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

        # 聚合门控 (Aggregation Gate)
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
    """兼容多数据集的模态专属编码器"""

    def __init__(self, dataset, modal_type):
        super().__init__()
        if dataset in ["MNIST", "CIFAR10"]:
            in_channels = 1 if dataset == "MNIST" else 3
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
            )
        else:  # SYSUMM01 (OSNet 前端降维)
            in_channels = 1 if modal_type == "IR" else 3
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

    def forward(self, x):
        return self.conv(x)


class SharedBackbone(nn.Module):
    """全局共享骨干网 (唯一参与 OTA 聚合)"""

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
        else:  # SYSUMM01 (核心 OSBlock 提取高层语义)
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

    def forward(self, x):
        return self.net(self.fdp(x))


class ClassifierHead(nn.Module):
    """个性化分类头 (留存本地)"""

    def __init__(self, dataset, num_classes):
        super().__init__()
        if dataset in ["MNIST", "CIFAR10"]:
            self.fc = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        else:  # SYSUMM01
            self.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.fc(x)