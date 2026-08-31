import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEncoder(nn.Module):
    """自适应轻量级视觉特征提取器 (支持 32x32 和 64x64 输入)"""
    def __init__(self, in_channels=3, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.proj(feat)


class TextEncoder(nn.Module):
    """自适应轻量级文本特征提取器 (Token序列 -> 128维特征)"""
    def __init__(self, vocab_size=1000, embed_dim=64, out_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        embed = self.embedding(x).transpose(1, 2)
        feat = self.conv1d(embed).view(embed.size(0), -1)
        return self.proj(feat)


class MultimodalFederatedModel(nn.Module):
    """多模态联合模型：视觉骨干 + 文本骨干 + 融合分类头"""
    def __init__(self, num_classes=100, vocab_size=1000, feature_dim=128):
        super().__init__()
        self.visual_encoder = VisualEncoder(in_channels=3, out_dim=feature_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=64, out_dim=feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, inputs):
        images, texts = inputs
        v_feat = self.visual_encoder(images)
        t_feat = self.text_encoder(texts)
        fusion_feat = torch.cat([v_feat, t_feat], dim=-1)
        logits = self.classifier(fusion_feat)
        return logits


class FederatedModel(nn.Module):
    """统一模型入口"""
    def __init__(self, dataset="MultimodalCIFAR100", num_classes=100):
        super().__init__()
        vocab_size = 1000 if "TinyImageNet" in dataset else 500
        self.model = MultimodalFederatedModel(num_classes=num_classes, vocab_size=vocab_size)

    def forward(self, inputs):
        return self.model(inputs)