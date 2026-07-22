import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# FashionMNIST：FedAvg 论文风格轻量 CNN（1通道输入）
# ============================================================

class FashionMNISTNet(nn.Module):
    """FedAvg 论文风格 CNN：2层Conv + 2层FC，1通道灰度输入"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=0),   # 28→24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 24→12
            nn.Conv2d(32, 64, kernel_size=5, padding=0),   # 12→8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 8→4
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# CIFAR10：修改版 ResNet-18（适配 32×32 小图像）
# ============================================================

class BasicBlock(nn.Module):
    """ResNet BasicBlock：2层 3×3 卷积 + 残差连接"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class CIFARResNet18(nn.Module):
    """修改版 ResNet-18：conv1=3×3/stride=1，删除 MaxPool，适配 32×32 输入"""

    def __init__(self, num_classes=10):
        super().__init__()
        # 第一层：3×3卷积，stride=1，保留分辨率
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 删除 MaxPool！32×32 图像不能过早下采样

        # 4个残差组：64→128→256→512
        self.layer1 = self._make_layer(64, 64, 2, stride=1)   # 32×32
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 32→16
        self.layer3 = self._make_layer(128, 256, 2, stride=2) # 16→8
        self.layer4 = self._make_layer(256, 512, 2, stride=2) # 8→4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * BasicBlock.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * BasicBlock.expansion),
            )
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 32×32
        x = self.layer1(x)                       # 32×32
        x = self.layer2(x)                       # 16×16
        x = self.layer3(x)                       # 8×8
        x = self.layer4(x)                       # 4×4
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================
# SYSU-MM01：OSNet 双流网络（GroupNorm, feat_dim=512）
# ============================================================

def _gn_groups(ch, max_groups=32):
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class ChannelAttention(nn.Module):
    """轻量通道注意力（SE模块）"""
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class OSBlock(nn.Module):
    """OSNet 全尺度残差块

    多个并行卷积流捕获不同尺度特征，通过通道注意力自适应融合。
    流1: 1×1（点尺度）
    流2: 1×1 → 3×3（小尺度）
    流3: 1×1 → 3×3 → 3×3（中尺度）
    流4: 1×1 → 3×3 → 3×3 → 3×3（大尺度）
    """

    def __init__(self, in_ch, out_ch, bottleneck_reduction=4, stride=1):
        super().__init__()
        mid = out_ch // bottleneck_reduction
        gn_mid = _gn_groups(mid)
        gn_out = _gn_groups(out_ch)

        # 流1: 点尺度（含stride下采样）
        self.stream1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, stride=stride, bias=False),
            nn.GroupNorm(gn_mid, mid), nn.ReLU(inplace=True),
        )
        # 流2: 小尺度
        self.stream2 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.GroupNorm(gn_mid, mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(gn_mid, mid), nn.ReLU(inplace=True),
        )
        # 流3: 中尺度
        self.stream3 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.GroupNorm(gn_mid, mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(gn_mid, mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.GroupNorm(gn_mid, mid), nn.ReLU(inplace=True),
        )
        # 流4: 大尺度
        self.stream4 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.GroupNorm(gn_mid, mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(gn_mid, mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.GroupNorm(gn_mid, mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.GroupNorm(gn_mid, mid), nn.ReLU(inplace=True),
        )

        # 融合：四流拼接 → 1×1压缩 + 通道注意力
        self.attention = ChannelAttention(out_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 4, out_ch, 1, bias=False),
            nn.GroupNorm(gn_out, out_ch),
        )

        # 下采样（stride>1 或通道不匹配时）
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(gn_out, out_ch),
            )

        self.stride = stride

    def forward(self, x):
        identity = x
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        out = torch.cat([s1, s2, s3, s4], dim=1)
        out = self.fuse(out)
        out = self.attention(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(out + identity, inplace=True)


class OSNetBackbone(nn.Module):
    """OSNet 共享骨干（轻量版，~2.2M参数）

    输入 32ch, 128×128 → AdaptiveAvgPool → 512维
    """

    def __init__(self, feat_dim=512):
        super().__init__()
        # Conv1: 32→64, stride=2 (128→64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_gn_groups(64), 64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)  # 64→32

        # OSBlock 层
        self.layer1 = self._make_layer(64, 256, blocks=2, stride=2)   # 32→16
        self.layer2 = self._make_layer(256, 384, blocks=2, stride=2)  # 16→8
        self.layer3 = self._make_layer(384, 512, blocks=2, stride=1)  # 8→8

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feat_dim)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [OSBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(OSBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ModalSpecificEncoder(nn.Module):
    """模态专属编码器：RGB/IR 各自独立的浅层编码（GroupNorm）"""

    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(32, 32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class DualStreamReID(nn.Module):
    """OSNet 双流跨模态行人重识别网络

    三段式：Modal-specific Encoder → OSNet Shared Backbone → BNNeck + Classifier
    ~2.5M参数，feat_dim=512，全部GroupNorm
    """

    def __init__(self, num_classes=395, feat_dim=512):
        super().__init__()
        self.rgb_encoder = ModalSpecificEncoder(in_channels=3)
        self.ir_encoder = ModalSpecificEncoder(in_channels=3)
        self.backbone = OSNetBackbone(feat_dim=feat_dim)
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x, modality='RGB'):
        if modality == 'RGB':
            x = self.rgb_encoder(x)
        else:
            x = self.ir_encoder(x)
        feat = self.backbone(x)
        feat = self.bnneck(feat)
        logits = self.classifier(feat)
        return logits, feat


# ============================================================
# 工厂函数（接口保持一致）
# ============================================================

def create_model(dataset_name, num_classes=10):
    """根据数据集创建模型（接口不变）"""
    if dataset_name == "MNIST":
        return FashionMNISTNet(num_classes)
    elif dataset_name in ("CIFAR10", "CIFAR100"):
        return CIFARResNet18(num_classes)
    elif dataset_name == "SYSU-MM01":
        return DualStreamReID(num_classes)
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
