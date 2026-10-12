import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        num_groups = 8 if out_channels % 8 == 0 else 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.relu(out + residual)
        return out

class ResBlock1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        num_groups = 8 if channels % 8 == 0 else 4
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.relu(out + residual)
        return out

class AerialVisualEncoder(nn.Module):
    """无人机高空俯拍摄像头轻量视觉提取器 (ResNet-10 with GroupNorm)"""
    def __init__(self, in_channels=3, out_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResBlock2d(32, 32, stride=1)
        self.layer2 = ResBlock2d(32, 64, stride=2)
        self.layer3 = ResBlock2d(64, 128, stride=2)
        self.layer4 = ResBlock2d(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        feat = self.stem(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        feat = self.pool(feat).view(feat.size(0), -1)
        return self.proj(feat)

class TelemetryTextEncoder(nn.Module):
    """无人机机载遥测/侦察日志文本提取器"""
    def __init__(self, vocab_size=2500, embed_dim=64, out_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResBlock1d(128)
        self.layer2 = ResBlock1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        embed = self.embedding(x).transpose(1, 2)
        feat = self.stem(embed)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.pool(feat).view(feat.size(0), -1)
        return self.proj(feat)

class AerialDualStreamNet(nn.Module):
    """无人机端到端双流地貌解译模型"""
    def __init__(self, num_classes=21, vocab_size=2500, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.visual_encoder = AerialVisualEncoder(in_channels=3, out_dim=feature_dim)
        self.text_encoder = TelemetryTextEncoder(vocab_size=vocab_size, embed_dim=64, out_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, inputs, drop_visual=False):
        images, texts = inputs
        if drop_visual:
            batch_size = texts.size(0)
            v_feat = torch.zeros(batch_size, self.feature_dim, device=texts.device, dtype=torch.float32)
        else:
            v_feat = self.visual_encoder(images)

        t_feat = self.text_encoder(texts)
        fusion_feat = torch.cat([v_feat, t_feat], dim=-1)
        logits = self.classifier(fusion_feat)
        return logits

class FederatedModel(nn.Module):
    def __init__(self, dataset="UAV-AER", num_classes=21):
        super().__init__()
        self.model = AerialDualStreamNet(num_classes=num_classes, vocab_size=2500)

    def forward(self, inputs, drop_visual=False):
        return self.model(inputs, drop_visual=drop_visual)