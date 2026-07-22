import numpy as np


class EdgeSBS:
    """边缘小基站(SBS)：管理UAV集群，支持空中计算聚合"""

    def __init__(self, sbs_id, config):
        self.id = sbs_id
        self.config = config

    def air_computing_aggregate(self, uav_gradients):
        """空中计算聚合：求和（与DMFL edge server一致）"""
        if isinstance(uav_gradients, np.ndarray) and uav_gradients.ndim == 2:
            return np.nansum(uav_gradients, axis=0)

        valid_grads = [g for g in uav_gradients if g is not None and len(g) > 0]
        if not valid_grads:
            return np.zeros(self.config.get('grad_dim', 1))
        return np.nansum(valid_grads, axis=0)
