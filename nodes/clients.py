import math
import numpy as np
import torch
import torch.nn as nn
from nodes.models import FederatedModel

class ClientUE:
    """UAV 边缘终端节点 (具备物理位置、信道衰落与计算建模)"""
    def __init__(self, client_id, config, train_loader, test_loader, num_classes):
        self.id = client_id
        self.config = config
        self.device = config["device"]
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = FederatedModel(config.get("dataset", "MultimodalFlowers102"), num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # 方案 B: 物理属性在初始化时静态固定 (依托主种子生成)
        clients_per_sbs = config.get("clients_per_sbs", 10)
        intra_idx = self.id % clients_per_sbs
        ratio = intra_idx / max(1, clients_per_sbs - 1)

        # 1. 静态几何距离: 100m 到 500m
        self.distance = 100.0 + 400.0 * ratio
        # 2. 静态计算主频: 1.0 GHz 到 2.5 GHz
        self.cpu_freq = 1.0e9 + 1.5e9 * (1.0 - ratio)
        # 3. 基础发射功率与带宽
        self.bandwidth = 1.0e6      # 1 MHz
        self.tx_power = 0.2         # 200 mW
        self.noise_power = 1e-13    # 噪声功率谱密度

    @property
    def num_train_samples(self):
        return len(self.train_loader.dataset)

    def _make_optimizer(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.get("local_lr", 0.01),
            momentum=self.config.get("momentum", 0.9),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

    def set_global_features(self, global_state):
        self.model.load_state_dict(global_state, strict=True)

    def simulate_communication_time(self, round_idx=0, drop_visual=False):
        """
        方案 B: 静态大尺度路径损耗 + 瞬时瑞利快衰落微扰
        drop_visual=True 时计算量和上传参数量大幅削减
        """
        # 统一受控随机种子，保证跨算法在相同轮次下信道微扰严格一致
        rng = np.random.default_rng(self.config.get("seed", 42) + 1000003 * round_idx + 9176 * self.id)
        fast_fading = rng.exponential(scale=1.0)

        # 路径损耗 PL = (c / (4*pi*f*d))^2
        path_loss = (3e8 / (4.0 * math.pi * 2.4e9 * self.distance)) ** 2
        h_channel = path_loss * fast_fading
        snr = (self.tx_power * h_channel) / self.noise_power
        uplink_rate = max(self.bandwidth * math.log2(1.0 + snr), 5e4)  # bps

        # 计算时延与传输时延
        cycles_per_sample = 150
        num_epochs = self.config.get("local_epochs", 4)
        if drop_visual:
            # 切断视觉分支后算力消耗降低 80%
            t_cpu = (num_epochs * (cycles_per_sample * 0.2) * self.num_train_samples) / self.cpu_freq
            # 仅传输文本和分类头参数
            param_count = sum(p.numel() for name, p in self.model.named_parameters() if 'visual' not in name)
        else:
            t_cpu = (num_epochs * cycles_per_sample * self.num_train_samples) / self.cpu_freq
            param_count = sum(p.numel() for p in self.model.parameters())

        t_up = (param_count * 32.0) / uplink_rate
        total_time = t_cpu + t_up
        return False, total_time

    def local_train(self, global_state, drop_visual=False):
        self.set_global_features(global_state)
        initial = {name: value.detach().clone() for name, value in global_state.items()}

        self.model.train()
        optimizer = self._make_optimizer()
        num_epochs = self.config.get("local_epochs", 4)

        for _ in range(num_epochs):
            for (images, texts), labels in self.train_loader:
                if len(labels) < 2:
                    continue
                texts = texts.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                images = images.to(self.device, non_blocking=True) if not drop_visual else images

                optimizer.zero_grad(set_to_none=True)
                logits = self.model((images, texts), drop_visual=drop_visual)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

        current = self.model.state_dict()
        updates = []
        head_split_idx = 0
        head_found = False

        for name, value in initial.items():
            if value.is_floating_point():
                is_head = ('classifier' in name or 'fc' in name)
                if is_head and not head_found:
                    head_split_idx = sum(u.size for u in updates)
                    head_found = True
                if drop_visual and 'visual' in name:
                    # 视觉分支未计算，更新量置零
                    updates.append(np.zeros(value.numel(), dtype=np.float32))
                else:
                    diff = (value.float() - current[name].float()).cpu().numpy().ravel()
                    updates.append(diff)
            else:
                updates.append(np.zeros(value.numel(), dtype=np.float32))

        grad = np.concatenate(updates).astype(np.float32, copy=False)
        if not head_found:
            head_split_idx = grad.size
        return grad, head_split_idx