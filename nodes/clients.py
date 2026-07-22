import numpy as np
import torch
import torch.nn as nn
from nodes.models import create_model


class ClientUAV:
    """联邦学习无人机客户端"""

    def __init__(self, client_id, config, train_loader, num_classes):
        self.id = client_id
        self.config = config
        self.device = config['device']
        self.train_loader = train_loader

        self.model = create_model(config['dataset'], num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.channel_quality = 1.0
        self.is_straggler_state = False
        self.straggler_reason = None

    def refresh_state(self, round_idx=0):
        self.channel_quality = np.random.uniform(0.1, 1.0)
        self.is_straggler_state = False
        self.straggler_reason = None
        if np.random.rand() < self.config['straggler_rate']:
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
        self.model.load_state_dict(global_state_dict, strict=False)

    def local_train(self, global_state_dict):
        self.set_global_model(global_state_dict)
        initial_params = {k: v.data.clone() for k, v in self.model.named_parameters()}

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=self.config['local_lr'], momentum=0.5)

        self.model.train()
        for _ in range(self.config['local_epochs']):
            for images, labels in self.train_loader:
                if len(images) == 0:
                    continue
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

        flat_grads = []
        for k, p in self.model.named_parameters():
            flat_grads.append((initial_params[k] - p.data).cpu().numpy().flatten())
        return np.concatenate(flat_grads)

    def local_train_return_state_dict(self, global_state_dict, frozen_keys=None):
        self.set_global_model(global_state_dict)

        frozen_params = []
        if frozen_keys:
            for name, param in self.model.named_parameters():
                if name in frozen_keys:
                    param.requires_grad_(False)
                    frozen_params.append(param)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=self.config['local_lr'], momentum=0.5)

        self.model.train()
        for _ in range(self.config['local_epochs']):
            for images, labels in self.train_loader:
                if len(images) == 0:
                    continue
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

        for param in frozen_params:
            param.requires_grad_(True)

        return {k: v.data.clone() for k, v in self.model.named_parameters()}
