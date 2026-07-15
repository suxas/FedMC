import numpy as np


class CloudServer:
    """云端服务器：负责全局梯度聚合"""

    def __init__(self, config):
        self.config = config

    def global_aggregate(self, edge_gradients):
        """对边缘梯度取均值，过滤无效梯度"""
        valid_grads = [g for g in edge_gradients if g is not None and np.sum(np.abs(g)) > 0]
        if not valid_grads:
            return np.zeros(self.config.get('grad_dim', 1))
        return np.mean(valid_grads, axis=0)