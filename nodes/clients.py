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

        # 模型实例化
        in_channels = 3 if config['dataset'] == "CIFAR10" else 1
        feature_dim = 32 * 8 * 8 if config['dataset'] == "CIFAR10" else 32 * 7 * 7

        self.feature_extractor = FeatureExtractor(in_channels).to(self.device)
        # 个性化分类器头，永远留在本地
        self.classifier = ClassifierHead(feature_dim, num_classes).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.channel_quality = 1.0
        self.refresh_state()

    def refresh_state(self):
        self.channel_quality = np.random.uniform(0.1, 1.0)
        self.is_straggler_state = (np.random.rand() < self.config['straggler_rate'])

    def set_global_features(self, global_state_dict):
        self.feature_extractor.load_state_dict(global_state_dict)

    def local_train(self, global_state_dict):
        """执行本地 SGD 训练，返回特征提取层的伪梯度 (Pseudo-gradient)"""
        self.set_global_features(global_state_dict)

        # 记录初始权重，用于计算增量
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
                features = self.feature_extractor(images)
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 计算并展平特征提取层的更新量 (1D Numpy Array)
        flat_pseudo_grad = []
        for init_w, curr_w in zip(initial_weights, self.feature_extractor.parameters()):
            diff = (init_w - curr_w.data).cpu().numpy().flatten()
            flat_pseudo_grad.append(diff)

        return np.concatenate(flat_pseudo_grad)