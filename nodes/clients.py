import numpy as np
import torch
import torch.nn as nn
import copy
from nodes.models import ModalEncoder, SharedBackbone, ClassifierHead


class ClientUE:
    def __init__(self, client_id, config, train_loader, test_loader, num_classes, modal_type):
        self.id = client_id
        self.config = config
        self.device = config['device']
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.modal_type = modal_type

        self.encoder = ModalEncoder(config['dataset'], modal_type).to(self.device)
        self.shared_backbone = SharedBackbone(config['dataset']).to(self.device)
        self.classifier = ClassifierHead(config['dataset'], num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def set_global_features(self, global_backbone_state):
        self.shared_backbone.load_state_dict(global_backbone_state)

    def _train_epoch(self):
        optimizer = torch.optim.SGD(
            list(self.encoder.parameters()) +
            list(self.shared_backbone.parameters()) +
            list(self.classifier.parameters()),
            lr=self.config['local_lr'], momentum=0.5
        )
        self.encoder.train()
        self.shared_backbone.train()
        self.classifier.train()

        for epoch in range(self.config['local_epochs']):
            for img_1, img_2, labels in self.train_loader:
                # 保护机制：防止单个样本导致 BatchNorm 崩溃
                if len(labels) <= 1:
                    continue

                labels = labels.to(self.device)
                inputs = img_1.to(self.device) if self.modal_type == "RGB" else img_2.to(self.device)

                optimizer.zero_grad()
                outputs = self.classifier(self.shared_backbone(self.encoder(inputs)))
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def local_train(self, global_backbone_state):
        self.set_global_features(global_backbone_state)
        # 完整的 state_dict 提取，兼容 OSNet 的 BatchNorm
        initial_state = {k: v.clone() for k, v in self.shared_backbone.state_dict().items()}
        self._train_epoch()

        curr_state = self.shared_backbone.state_dict()
        flat_pseudo_grad = []
        for k in initial_state.keys():
            diff = (initial_state[k].float() - curr_state[k].float()).cpu().numpy().flatten()
            flat_pseudo_grad.append(diff)

        return np.concatenate(flat_pseudo_grad)

    def local_train_return_state_dict(self, global_backbone_state):
        self.set_global_features(global_backbone_state)
        self._train_epoch()
        return copy.deepcopy(self.shared_backbone.state_dict())