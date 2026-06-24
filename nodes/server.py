import numpy as np


class CloudServer:
    def __init__(self, config):
        self.config = config

    def global_aggregate(self, edge_gradients):
        """MBS 最基础的全局聚合行为 (加权平均)"""
        # 过滤掉完全无效的边缘梯度
        valid_grads = [g for g in edge_gradients if g is not None]
        if not valid_grads:
            return np.zeros(self.config.get('grad_dim', 500))
        return np.mean(valid_grads, axis=0)