import numpy as np
import torch
import torch.nn as nn
from nodes.models import create_feature_extractor, create_classifier


class ClientUAV:
    """联邦学习无人机客户端：负责本地训练"""

    def __init__(self, client_id, config, train_loader, num_classes):
        self.id = client_id
        self.config = config
        self.device = config['device']
        self.train_loader = train_loader

        # 创建本地模型
        dataset = config['dataset']
        self.feature_extractor = create_feature_extractor(dataset).to(self.device)
        feature_dim = self.feature_extractor.get_feature_dim()
        self.classifier = create_classifier(dataset, feature_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # 通信状态
        self.channel_quality = 1.0
        self.is_straggler_state = False
        self.straggler_reason = None

    def refresh_state(self, round_idx=0):
        """刷新信道质量和掉队状态，模拟UAV通信环境"""
        self.channel_quality = np.random.uniform(0.1, 1.0)

        self.is_straggler_state = False
        self.straggler_reason = None
        r = np.random.rand()

        if r < self.config['straggler_rate']:
            self.is_straggler_state = True
            reason_roll = np.random.rand()
            if reason_roll < 0.4:
                self.straggler_reason = "timeout"
            elif reason_roll < 0.7:
                self.straggler_reason = "out_of_range"
            elif reason_roll < 0.85:
                self.straggler_reason = "battery"
            else:
                self.straggler_reason = "hardware"

        if self.channel_quality < self.config['channel_threshold']:
            self.is_straggler_state = True
            self.straggler_reason = "poor_channel"

    def set_global_model(self, global_state_dict):
        """加载全局模型权重"""
        fe_keys = [k for k, _ in self.feature_extractor.named_parameters()]
        fe_sd = {k: global_state_dict[f'feature_extractor.{k}'] for k in fe_keys}
        self.feature_extractor.load_state_dict(fe_sd, strict=False)

        cl_keys = [k for k, _ in self.classifier.named_parameters()]
        cl_sd = {k: global_state_dict[f'classifier.{k}'] for k in cl_keys}
        self.classifier.load_state_dict(cl_sd, strict=False)

    def local_train(self, global_state_dict):
        """本地SGD训练，返回伪梯度（初始权重-训练后权重）"""
        self.set_global_model(global_state_dict)

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

        flat_grads = []
        for init_w, p in zip(initial_fe, self.feature_extractor.parameters()):
            flat_grads.append((init_w - p.data).cpu().numpy().flatten())
        for init_w, p in zip(initial_cl, self.classifier.parameters()):
            flat_grads.append((init_w - p.data).cpu().numpy().flatten())
        return np.concatenate(flat_grads)

    def local_train_return_state_dict(self, global_state_dict, frozen_keys=None):
        """本地训练后返回state_dict（供SALF使用）

        Args:
            global_state_dict: 全局模型权重
            frozen_keys: 需要冻结的层key集合（掉队者不更新这些层）
        """
        self.set_global_model(global_state_dict)

        # 冻结指定层：requires_grad=False，SGD不会更新这些参数
        frozen_params = []
        if frozen_keys:
            for name, param in self.feature_extractor.named_parameters():
                if f'feature_extractor.{name}' in frozen_keys:
                    param.requires_grad_(False)
                    frozen_params.append(param)
            for name, param in self.classifier.named_parameters():
                if f'classifier.{name}' in frozen_keys:
                    param.requires_grad_(False)
                    frozen_params.append(param)

        # 只训练requires_grad=True的参数
        trainable_params = [p for p in
                            list(self.feature_extractor.parameters()) + list(self.classifier.parameters())
                            if p.requires_grad]

        optimizer = torch.optim.SGD(trainable_params, lr=self.config['local_lr'], momentum=0.5)

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

        # 恢复requires_grad
        for param in frozen_params:
            param.requires_grad_(True)

        full_sd = {}
        for k, v in self.feature_extractor.named_parameters():
            full_sd[f'feature_extractor.{k}'] = v.data.clone()
        for k, v in self.classifier.named_parameters():
            full_sd[f'classifier.{k}'] = v.data.clone()
        return full_sd
