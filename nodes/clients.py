import copy
import numpy as np
import torch
import torch.nn as nn
from nodes.models import create_feature_extractor, create_classifier


class ClientUE:
    """联邦学习客户端：负责本地训练，支持两种训练模式（梯度/状态字典）"""

    def __init__(self, client_id, config, train_loader, num_classes, modality=None):
        self.id = client_id
        self.config = config
        self.device = config['device']
        self.train_loader = train_loader

        # 模态类型：如果指定了则使用，否则随机分配
        if modality is not None:
            self.modality = modality
        else:
            self.modality = 'RGB' if np.random.rand() < config['modality_rgb_ratio'] else 'NIR'

        # 根据数据集确定输入通道数
        dataset = config['dataset']
        if dataset in ("SYSU-MM01", "CIFAR10"):
            in_channels = 3
        else:
            in_channels = 1

        # 创建本地模型
        self.feature_extractor = create_feature_extractor(dataset, in_channels).to(self.device)
        feature_dim = self.feature_extractor.get_feature_dim()
        self.classifier = create_classifier(dataset, feature_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.channel_quality = 1.0
        self.is_straggler_state = False

    def refresh_state(self):
        """刷新信道质量和掉队状态（每轮通信前调用）"""
        self.channel_quality = np.random.uniform(0.1, 1.0)
        self.is_straggler_state = np.random.rand() < self.config['straggler_rate']

    def set_global_model(self, global_state_dict):
        """加载全局模型权重（feature_extractor + classifier）"""
        fe_keys = [k for k, _ in self.feature_extractor.named_parameters()]
        fe_sd = {k: global_state_dict[f'feature_extractor.{k}'] for k in fe_keys}
        self.feature_extractor.load_state_dict(fe_sd, strict=False)

        cl_keys = [k for k, _ in self.classifier.named_parameters()]
        cl_sd = {k: global_state_dict[f'classifier.{k}'] for k in cl_keys}
        self.classifier.load_state_dict(cl_sd, strict=False)

    def local_train(self, global_state_dict):
        """本地SGD训练，返回feature_extractor + classifier的伪梯度（初始权重-训练后权重）"""
        self.set_global_model(global_state_dict)

        # 记录初始权重（feature_extractor + classifier）
        initial_fe = [p.data.clone() for p in self.feature_extractor.parameters()]
        initial_cl = [p.data.clone() for p in self.classifier.parameters()]

        optimizer = torch.optim.SGD(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=self.config['local_lr'], momentum=0.5
        )

        self.feature_extractor.train()
        self.classifier.train()

        for _ in range(self.config['local_epochs']):
            for images, labels in self.train_loader:
                if len(images) == 0:
                    continue
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.classifier(self.feature_extractor(images)), labels)
                loss.backward()
                optimizer.step()

        # 计算伪梯度：feature_extractor + classifier
        flat_grads = []
        for init_w, p in zip(initial_fe, self.feature_extractor.parameters()):
            flat_grads.append((init_w - p.data).cpu().numpy().flatten())
        for init_w, p in zip(initial_cl, self.classifier.parameters()):
            flat_grads.append((init_w - p.data).cpu().numpy().flatten())
        return np.concatenate(flat_grads)

    def local_train_return_state_dict(self, global_state_dict):
        """本地训练后返回完整state_dict（feature_extractor + classifier，供SALF使用）"""
        self.set_global_model(global_state_dict)

        optimizer = torch.optim.SGD(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=self.config['local_lr'], momentum=0.5
        )

        self.feature_extractor.train()
        self.classifier.train()

        for _ in range(self.config['local_epochs']):
            for images, labels in self.train_loader:
                if len(images) == 0:
                    continue
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.classifier(self.feature_extractor(images)), labels)
                loss.backward()
                optimizer.step()

        # 合并 feature_extractor + classifier 的参数（仅nn.Parameter，不含buffer）
        full_sd = {}
        for k, v in self.feature_extractor.named_parameters():
            full_sd[f'feature_extractor.{k}'] = v.data.clone()
        for k, v in self.classifier.named_parameters():
            full_sd[f'classifier.{k}'] = v.data.clone()
        return full_sd
