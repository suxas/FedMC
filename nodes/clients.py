import copy
import random
import numpy as np
import torch
import torch.nn as nn
from nodes.models import FederatedModel


class ClientUE:
    """联邦学习边缘客户端 (UAV)"""

    def __init__(self, client_id, config, train_loader, test_loader, num_classes):
        self.id = client_id
        self.config = config
        self.device = config["device"]
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = FederatedModel(config["dataset"], num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    @property
    def num_train_samples(self):
        return len(self.train_loader.dataset)

    def _make_optimizer(self):
        # 固定 0.01 学习率，集成 weight_decay 抑制高维参数漂移
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.get("local_lr", 0.01),
            momentum=self.config.get("momentum", 0.9),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

    def set_global_features(self, global_state):
        self.model.load_state_dict(global_state, strict=True)

    def _train_epoch(self):
        self.model.train()
        optimizer = self._make_optimizer()
        total_loss = 0.0
        n_batches = 0
        for _ in range(self.config.get("local_epochs", 3)):
            for images, labels in self.train_loader:
                if len(labels) < 2:
                    continue
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def local_train(self, global_state):
        self.set_global_features(global_state)
        initial = {name: value.detach().clone() for name, value in global_state.items()}
        self._train_epoch()
        current = self.model.state_dict()

        updates = []
        head_split_idx = 0
        head_found = False

        for name, value in initial.items():
            if value.is_floating_point():
                is_head = ('fc' in name or 'classifier' in name)
                if is_head and not head_found:
                    head_split_idx = sum(u.size for u in updates)
                    head_found = True
                diff = (value.float() - current[name].float()).cpu().numpy().ravel()
                updates.append(diff)
            else:
                updates.append(np.zeros(value.numel(), dtype=np.float32))

        grad = np.concatenate(updates).astype(np.float32, copy=False)
        if not head_found:
            head_split_idx = grad.size
        return grad, head_split_idx

    def simulate_communication_time(self, round_idx=0):
        rng = random.Random(self.config.get("seed", 42) + 1000003 * round_idx + 9176 * self.id)
        freq = rng.uniform(1e9, 2e9)
        rate = rng.uniform(1e6, 5e6)

        cycles_per_sample = 100
        n_samples = self.num_train_samples
        num_local_epochs = self.config.get("local_epochs", 3)
        t_cpu = (num_local_epochs * cycles_per_sample * n_samples) / freq

        param_size = sum(p.numel() for p in self.model.parameters())
        t_up = (param_size * 32) / rate

        return False, (t_cpu + t_up)