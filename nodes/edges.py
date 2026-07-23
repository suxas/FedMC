import numpy as np
import warnings


class EdgeSBS:
    """边缘小基站 (SBS)：管理 UAV 集群，支持 LRMC 梯度补全和缓冲池"""

    def __init__(self, sbs_id, config):
        self.id = sbs_id
        self.config = config
        self.buffer_size = config.get('buffer_size', 5)
        self.compensation_weight = config.get('compensation_weight', 0.7)
        self.history_weight = config.get('history_weight', 0.3)
        # 缓冲池：存储每个客户端的历史梯度 {client_id: [grad_t, grad_t-1, ...]}
        self.gradient_buffer = {}

    def collect_gradients(self, clients, global_state_dict):
        """收集客户端梯度，模拟掉队检测

        Returns:
            grad_matrix: [num_clients × grad_dim]，掉队者对应行为 NaN
            straggler_ids: 掉队客户端 ID 列表
            normal_ids: 正常客户端 ID 列表
        """
        grads = []
        straggler_ids = []
        normal_ids = []

        for client in clients:
            # 模拟掉队判定
            is_straggler, comm_time = client.simulate_communication_time()

            if is_straggler:
                straggler_ids.append(client.id)
                grads.append(None)
            else:
                grad = client.local_train(global_state_dict)
                grads.append(grad)
                normal_ids.append(client.id)
                # 更新缓冲池
                if client.id not in self.gradient_buffer:
                    self.gradient_buffer[client.id] = []
                self.gradient_buffer[client.id].append(grad.copy())
                if len(self.gradient_buffer[client.id]) > self.buffer_size:
                    self.gradient_buffer[client.id].pop(0)

        # 构建梯度矩阵
        grad_dim = len(grads[normal_ids[0] - clients[0].id]) if normal_ids else 0
        if grad_dim == 0:
            # 所有客户端都掉队
            grad_dim = self.config.get('grad_dim', 1)

        grad_matrix = np.full((len(clients), grad_dim), np.nan)
        for i, g in enumerate(grads):
            if g is not None:
                grad_matrix[i, :len(g)] = g

        return grad_matrix, straggler_ids, normal_ids

    def lrmc_complete(self, grad_matrix):
        """LRMC 梯度补全：迭代式 SVD 降秩补全

        Args:
            grad_matrix: [num_clients × grad_dim]，NaN 表示缺失

        Returns:
            completed_matrix: 补全后的梯度矩阵
        """
        max_iter = self.config.get('lrmc_max_iter', 50)
        tol = self.config.get('lrmc_tol', 1e-4)
        rank_ratio = self.config.get('lrmc_rank_ratio', 0.5)

        if np.isnan(grad_matrix).all():
            return np.zeros_like(grad_matrix)

        # 记录缺失位置
        missing_mask = np.isnan(grad_matrix)

        # Step 1: 列均值填充缺失值
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_mean = np.nanmean(grad_matrix, axis=0)
        col_mean = np.nan_to_num(col_mean, nan=0.0)

        mat_filled = np.copy(grad_matrix)
        for i in range(mat_filled.shape[0]):
            missing_cols = np.where(missing_mask[i])[0]
            mat_filled[i, missing_cols] = col_mean[missing_cols]

        # Step 2: 迭代 SVD 降秩补全
        for iteration in range(max_iter):
            mat_old = mat_filled.copy()

            # SVD 分解
            try:
                U, s, Vt = np.linalg.svd(mat_filled, full_matrices=False)
            except np.linalg.LinAlgError:
                break

            # 自适应秩选择：保留 rank_ratio 的能量
            total_energy = np.sum(s ** 2)
            cumulative_energy = np.cumsum(s ** 2)
            rank = max(1, np.searchsorted(cumulative_energy, total_energy * rank_ratio) + 1)
            rank = min(rank, len(s))

            # 截断奇异值
            s_truncated = np.zeros_like(s)
            s_truncated[:rank] = s[:rank]

            # 重建
            mat_reconstructed = U @ np.diag(s_truncated) @ Vt

            # 只更新缺失位置
            mat_filled[missing_mask] = mat_reconstructed[missing_mask]

            # 收敛检查
            diff = np.linalg.norm(mat_filled - mat_old) / (np.linalg.norm(mat_old) + 1e-10)
            if diff < tol:
                break

        return mat_filled

    def get_compensated_gradient(self, client_id, lrmc_grad):
        """使用缓冲池对掉队者梯度进行补偿

        补偿梯度 = compensation_weight × LRMC补全梯度 + history_weight × 历史梯度均值
        """
        if client_id in self.gradient_buffer and len(self.gradient_buffer[client_id]) > 0:
            history_grad = np.mean(self.gradient_buffer[client_id], axis=0)
            compensated = (self.compensation_weight * lrmc_grad +
                           self.history_weight * history_grad)
            return compensated
        else:
            # 无历史梯度，直接使用 LRMC 补全结果
            return lrmc_grad

    def intra_cluster_aggregate(self, grad_matrix, straggler_ids, clients):
        """簇内聚合：对补全后的梯度矩阵做加权平均

        Args:
            grad_matrix: 补全后的梯度矩阵 [num_clients × grad_dim]
            straggler_ids: 掉队客户端 ID 列表
            clients: 客户端列表

        Returns:
            aggregated_grad: 聚合后的梯度向量 [grad_dim]
        """
        # 对掉队者的 LRMC 补全梯度进行缓冲池补偿
        for i, client in enumerate(clients):
            if client.id in straggler_ids:
                grad_matrix[i] = self.get_compensated_gradient(client.id, grad_matrix[i])

        # 加权平均（等权）
        return np.mean(grad_matrix, axis=0)

    def ota_to_lbs(self, aggregated_grad):
        """OTA 传输到大基站（可选：添加信道噪声）"""
        snr_db = self.config.get('ota_snr_db', 30.0)
        if snr_db <= 0:
            return aggregated_grad

        signal_power = np.mean(aggregated_grad ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear if snr_linear > 0 else 0
        noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-10)), aggregated_grad.shape)
        return aggregated_grad + noise
