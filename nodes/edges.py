import numpy as np


class EdgeSBS:
    """边缘小基站(SBS)：管理UAV集群，支持空中计算聚合"""

    def __init__(self, sbs_id, config):
        self.id = sbs_id
        self.config = config

    def air_computing_aggregate(self, uav_gradients, uav_weights=None):
        """空中计算聚合（Air Computing）

        利用无线信道叠加特性，所有UAV并发传输梯度，SBS在空口直接接收叠加信号。
        等效于对所有UAV梯度做加权求和（信道自然实现加权叠加）。

        Args:
            uav_gradients: list of 1D gradient arrays
            uav_weights: 各UAV的权重（如数据量），默认均等权重

        Returns:
            聚合后的梯度（加权平均）
        """
        valid_grads = [g for g in uav_gradients if g is not None and len(g) > 0]
        if not valid_grads:
            return np.zeros(self.config.get('grad_dim', 1))

        if uav_weights is None:
            uav_weights = [1.0] * len(valid_grads)

        total_weight = sum(uav_weights[:len(valid_grads)])
        aggregated = np.zeros_like(valid_grads[0])
        for g, w in zip(valid_grads, uav_weights[:len(valid_grads)]):
            aggregated += w * g
        aggregated /= total_weight

        return aggregated

    def standard_aggregate(self, uav_gradients):
        """常规通信聚合：简单均值"""
        valid_grads = [g for g in uav_gradients if g is not None and len(g) > 0]
        if not valid_grads:
            return np.zeros(self.config.get('grad_dim', 1))
        return np.mean(valid_grads, axis=0)
