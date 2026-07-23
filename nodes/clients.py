import numpy as np
import torch
import torch.nn as nn
import copy
import random
from nodes.models import ModalEncoder, SharedBackbone, ClassifierHead, DualModalFusion


class ClientUE:
    """联邦学习客户端（UAV），支持 RGB / IR / Both 三种模态"""

    def __init__(self, client_id, config, train_loader, test_loader, num_classes, modal_type):
        self.id = client_id
        self.config = config
        self.device = config['device']
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.modal_type = modal_type  # "RGB", "IR", "Both"

        dataset = config['dataset']

        # 构建编码器
        if modal_type == "Both":
            self.encoder_rgb = ModalEncoder(dataset, "RGB").to(self.device)
            self.encoder_ir = ModalEncoder(dataset, "IR").to(self.device)
            self.fusion = DualModalFusion(self.encoder_rgb.out_channels).to(self.device)
            self.encoder = None  # 不使用单一编码器
        elif modal_type == "RGB":
            self.encoder = ModalEncoder(dataset, "RGB").to(self.device)
            self.encoder_rgb = self.encoder
            self.encoder_ir = None
            self.fusion = None
        else:  # IR
            self.encoder = ModalEncoder(dataset, "IR").to(self.device)
            self.encoder_rgb = None
            self.encoder_ir = self.encoder
            self.fusion = None

        self.shared_backbone = SharedBackbone(dataset).to(self.device)
        self.classifier = ClassifierHead(dataset, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def _get_all_local_params(self):
        """获取所有本地参数（编码器 + 骨干 + 分类头）"""
        params = []
        if self.encoder_rgb is not None:
            params.extend(self.encoder_rgb.parameters())
        if self.encoder_ir is not None and self.encoder_ir is not self.encoder_rgb:
            params.extend(self.encoder_ir.parameters())
        if self.fusion is not None:
            params.extend(self.fusion.parameters())
        params.extend(self.shared_backbone.parameters())
        params.extend(self.classifier.parameters())
        return params

    def _forward(self, img_rgb, img_ir):
        """前向传播，根据模态类型选择路径"""
        if self.modal_type == "Both":
            feat_rgb = self.encoder_rgb(img_rgb)
            feat_ir = self.encoder_ir(img_ir)
            feat = self.fusion(feat_rgb, feat_ir)
        elif self.modal_type == "RGB":
            feat = self.encoder(img_rgb)
        else:  # IR
            feat = self.encoder(img_ir)

        features = self.shared_backbone(feat)
        return self.classifier(features)

    def set_global_features(self, global_backbone_state):
        """加载全局骨干网权重（FedBN: 跳过 BN 统计量）"""
        local_state = self.shared_backbone.state_dict()
        for name, param in global_backbone_state.items():
            if 'running_mean' not in name and 'running_var' not in name and 'num_batches_tracked' not in name:
                local_state[name].copy_(param)
        self.shared_backbone.load_state_dict(local_state)

    def _train_epoch(self):
        """本地训练一个 epoch"""
        optimizer = torch.optim.SGD(
            self._get_all_local_params(),
            lr=self.config['local_lr'], momentum=0.5
        )

        # 设置训练模式
        if self.encoder_rgb is not None:
            self.encoder_rgb.train()
        if self.encoder_ir is not None and self.encoder_ir is not self.encoder_rgb:
            self.encoder_ir.train()
        if self.fusion is not None:
            self.fusion.train()
        self.shared_backbone.train()
        self.classifier.train()

        total_loss = 0.0
        n_batches = 0

        for epoch in range(self.config['local_epochs']):
            for img_1, img_2, labels in self.train_loader:
                if len(labels) <= 1:
                    continue

                labels = labels.to(self.device)
                img_rgb = img_1.to(self.device)
                img_ir = img_2.to(self.device)

                optimizer.zero_grad()
                outputs = self._forward(img_rgb, img_ir)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # 梯度裁剪
                if self.encoder_rgb is not None:
                    torch.nn.utils.clip_grad_norm_(self.encoder_rgb.parameters(), max_norm=5.0)
                if self.encoder_ir is not None and self.encoder_ir is not self.encoder_rgb:
                    torch.nn.utils.clip_grad_norm_(self.encoder_ir.parameters(), max_norm=5.0)
                if self.fusion is not None:
                    torch.nn.utils.clip_grad_norm_(self.fusion.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(self.shared_backbone.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=5.0)

                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def local_train(self, global_backbone_state):
        """本地训练并返回伪梯度（只返回 SharedBackbone 的梯度）"""
        self.set_global_features(global_backbone_state)
        initial_state = {k: v.clone() for k, v in self.shared_backbone.state_dict().items()}
        self._train_epoch()

        curr_state = self.shared_backbone.state_dict()
        flat_pseudo_grad = []
        for k in initial_state.keys():
            diff = (initial_state[k].float() - curr_state[k].float()).cpu().numpy().flatten()
            flat_pseudo_grad.append(diff)

        return np.concatenate(flat_pseudo_grad)

    def local_train_return_state_dict(self, global_backbone_state):
        """本地训练并返回 state_dict（SALF 用）"""
        self.set_global_features(global_backbone_state)
        self._train_epoch()
        return copy.deepcopy(self.shared_backbone.state_dict())

    def simulate_communication_time(self):
        """模拟计算时间 + 传输时延，返回是否掉队"""
        compute_time = max(0.1, random.gauss(
            self.config.get('compute_time_mean', 0.5),
            self.config.get('compute_time_std', 0.2)
        ))
        transmit_time = max(0.05, random.gauss(
            self.config.get('transmit_time_mean', 0.3),
            self.config.get('transmit_time_std', 0.1)
        ))
        total_time = compute_time + transmit_time
        is_straggler = total_time > self.config.get('time_window', 1.0)
        return is_straggler, total_time
