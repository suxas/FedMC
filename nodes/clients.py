import numpy as np


class ClientUE:
    def __init__(self, client_id, config):
        self.id = client_id
        self.config = config
        self.channel_quality = 1.0
        self.refresh_state()

    def refresh_state(self):
        """
        每轮刷新信道状态与系统算力状态。
        结合 config 中的 straggler_rate 模拟当前轮次该设备是否陷入算力或信道瓶颈。
        """
        self.channel_quality = np.random.uniform(0.1, 1.0)
        # 依据设定的掉队率，随机决定该设备本轮是否处于“劣势状态”
        self.is_straggler_state = (np.random.rand() < self.config['straggler_rate'])

    def local_train(self):
        """
        模拟本地特征层梯度的训练，体现 local_lr 和 local_epochs 的作用。
        """
        epochs = self.config.get('local_epochs', 1)
        lr = self.config.get('local_lr', 0.01)
        grad_dim = self.config.get('grad_dim', 500)

        accumulated_grad = np.zeros(grad_dim)

        # 模拟执行 local_epochs 轮本地 SGD
        for epoch in range(epochs):
            # 加入信道偏移模拟 Non-IID 数据带来的梯度差异
            step_gradient = np.random.normal(loc=self.channel_quality, scale=0.1, size=grad_dim)
            # 使用本地学习率累加梯度
            accumulated_grad += step_gradient * lr

        return accumulated_grad