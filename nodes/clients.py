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
        self.criterion = nn.CrossEntropyLoss()

    @property
    def num_train_samples(self):
        return len(self.train_loader.dataset)

    def _make_optimizer(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.get("local_lr", 0.001),
            momentum=self.config.get("momentum", 0.5),
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
        """模拟计算和上传时间，参考DMFL实现。
        
        返回计算+上传的总时间，由调用方决定是否掉队。
        使用随机CPU频率和上行速率来模拟异构性。
        """
        rng = random.Random(self.config.get("seed", 42) + 1000003 * round_idx + 9176 * self.id)
        
        # 随机CPU频率 [1GHz, 2GHz]（参考DMFL）
        freq = rng.uniform(1e9, 2e9)
        # 随机上行速率 [1Mbps, 5Mbps]（参考DMFL）
        rate = rng.uniform(1e6, 5e6)
        
        # 计算时间 = (cycles_per_sample * n_samples) / freq
        cycles_per_sample = 100
        n_samples = self.num_train_samples
        num_local_epochs = self.config.get("local_epochs", 3)
        t_cpu = (num_local_epochs * cycles_per_sample * n_samples) / freq
        
        # 上传时间 = (param_size * 32 bits) / rate
        param_size = 10000  # 约10K参数
        t_up = (param_size * 32) / rate
        
        total_time = t_cpu + t_up
        
        # 返回时间，由调用方决定是否掉队
        # 掉队判断基于同SBS内客户端的时间排序（参考DMFL）
        return False, total_time
