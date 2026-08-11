import copy
import random

import numpy as np
import torch
import torch.nn as nn

from nodes.models import FederatedModel


class ClientUE:
    """Federated client that trains a synchronized model."""

    def __init__(self, client_id, config, train_loader, test_loader, num_classes):
        self.id = client_id
        self.config = config
        self.device = config["device"]
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = FederatedModel(config["dataset"], num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get("label_smoothing", 0.0)
        )

    @property
    def num_train_samples(self):
        return len(self.train_loader.dataset)

    def _make_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("local_lr", 3e-4),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

    def set_global_features(self, global_state):
        if set(global_state) != set(self.model.state_dict()):
            raise ValueError("Global and local model state keys differ")
        self.model.load_state_dict(global_state, strict=True)

    def set_eval_mode(self):
        self.model.eval()

    def _train_epoch(self):
        self.model.train()
        optimizer = self._make_optimizer()
        total_loss = 0.0
        n_batches = 0
        for _ in range(self.config.get("local_epochs", 1)):
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
        head_split_idx = 0  # 分类头梯度的起始位置
        head_found = False
        for name, value in initial.items():
            if value.is_floating_point():
                is_head = ('fc2' in name or 'classifier' in name)
                if is_head and not head_found:
                    head_split_idx = sum(u.size for u in updates)
                    head_found = True
                difference = value.float() - current[name].float()
                updates.append(difference.cpu().numpy().ravel())
            else:
                updates.append(np.zeros(value.numel(), dtype=np.float32))
        grad = np.concatenate(updates).astype(np.float32, copy=False)
        if not head_found:
            head_split_idx = grad.size
        return grad, head_split_idx

    def local_train_return_state_dict(self, global_state):
        self.set_global_features(global_state)
        self._train_epoch()
        return copy.deepcopy(self.model.state_dict())

    def simulate_communication_time(self, round_idx=0):
        rng = random.Random(self.config.get("seed", 42) + 1000003 * round_idx + 9176 * self.id)
        compute_time = max(0.1, rng.gauss(
            self.config.get("compute_time_mean", 0.5),
            self.config.get("compute_time_std", 0.2),
        ))
        transmit_time = max(0.05, rng.gauss(
            self.config.get("transmit_time_mean", 0.3),
            self.config.get("transmit_time_std", 0.1),
        ))
        ratio = min(max(self.config.get("straggler_ratio", 0.0), 0.0), 1.0)
        return rng.random() < ratio, compute_time + transmit_time
