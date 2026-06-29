import numpy as np


class CloudServer:
    def __init__(self, config):
        self.config = config

    def global_aggregate(self, edge_gradients):
        """对各边缘服务器的梯度做简单平均聚合"""
        valid_grads = [g for g in edge_gradients if g is not None]
        if not valid_grads:
            return np.zeros(self.config.get('grad_dim', 500))
        return np.mean(valid_grads, axis=0)
