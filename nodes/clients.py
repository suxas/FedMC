import copy  # 【修复点1】：引入 copy 模块供 deepcopy 使用
import numpy as np
import torch
import torch.nn as nn
from nodes.models import FeatureExtractor, ClassifierHead


class ClientUE:
    def __init__(self, client_id, config, train_loader, num_classes):
        self.id = client_id
        self.config = config
        self.device = config['device']
        self.train_loader = train_loader

        # 异构的多模态载荷属性
        self.modality = 'RGB' if np.random.rand() < config['modality_rgb_ratio'] else 'NIR'

        in_channels = 3 if config['dataset'] == "CIFAR10" else 1
        feature_dim = 32 * 8 * 8 if config['dataset'] == "CIFAR10" else 32 * 7 * 7

        self.feature_extractor = FeatureExtractor(in_channels).to(self.device)
        self.classifier = ClassifierHead(feature_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.channel_quality = 1.0
        self.is_straggler_state = False

    def refresh_state(self):
        """刷新节点的信道质量和物理受限情况"""
        self.channel_quality = np.random.uniform(0.1, 1.0)
        self.is_straggler_state = (np.random.rand() < self.config['straggler_rate'])

    def set_global_features(self, global_state_dict):
        """将云端下发的全局特征提取层权重加载到本地"""
        self.feature_extractor.load_state_dict(global_state_dict)

    def local_train(self, global_state_dict):
        """(供 FedAvg 和 Proposed 使用) 本地 SGD 训练，返回特征层的伪梯度(1D向量)"""
        self.set_global_features(global_state_dict)
        initial_weights = [p.data.clone() for p in self.feature_extractor.parameters()]

        optimizer = torch.optim.SGD(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=self.config['local_lr'], momentum=0.5
        )

        self.feature_extractor.train()
        self.classifier.train()

        for epoch in range(self.config['local_epochs']):
            for images, labels in self.train_loader:
                if len(images) == 0: continue
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                loss = self.criterion(self.classifier(self.feature_extractor(images)), labels)
                loss.backward()
                optimizer.step()

        flat_grads = []
        for init_w, curr_w in zip(initial_weights, self.feature_extractor.parameters()):
            flat_grads.append((init_w - curr_w.data).cpu().numpy().flatten())
        return np.concatenate(flat_grads)

    def local_train_return_state_dict(self, global_state_dict):
        """
        【修复点2】：供 SALF 算法使用的训练函数。
        本地训练后，直接返回训练好的特征提取器 state_dict（字典），方便按层替换。
        """
        self.set_global_features(global_state_dict)

        optimizer = torch.optim.SGD(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=self.config['local_lr'], momentum=0.5
        )

        self.feature_extractor.train()
        self.classifier.train()

        for epoch in range(self.config['local_epochs']):
            for images, labels in self.train_loader:
                if len(images) == 0: continue
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                features = self.feature_extractor(images)
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 使用深拷贝返回字典，防止后续在内存中被意外篡改
        return copy.deepcopy(self.feature_extractor.state_dict())