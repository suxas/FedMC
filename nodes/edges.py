import warnings

import numpy as np


class EdgeSBS:
    """Edge aggregation with robust LRMC and history compensation."""

    def __init__(self, sbs_id, config):
        self.id = sbs_id
        self.config = config
        self.buffer_size = config.get("buffer_size", 5)
        self.compensation_weight = config.get("compensation_weight", 0.7)
        self.gradient_buffer = {}
        self.grad_dim = config.get("grad_dim")
        self._row_client_ids = []
        self._head_split_idx = None  # 分类头梯度的起始位置

    def collect_gradients(self, clients, global_state_dict, round_idx=0):
        gradients, straggler_ids, normal_ids = [], [], []
        self._row_client_ids = [client.id for client in clients]
        head_split_idx = None
        for client in clients:
            is_straggler, _ = client.simulate_communication_time(round_idx)
            if is_straggler:
                gradients.append(None)
                straggler_ids.append(client.id)
                continue
            gradient, split_idx = client.local_train(global_state_dict)
            gradient = np.asarray(gradient).reshape(-1)
            if head_split_idx is None:
                head_split_idx = split_idx
            if not np.isfinite(gradient).all():
                raise ValueError(f"Client {client.id} produced a non-finite gradient")
            if self.grad_dim is None:
                self.grad_dim = gradient.size
            if gradient.size != self.grad_dim:
                raise ValueError(f"Client {client.id} gradient size mismatch")
            gradients.append(gradient)
            normal_ids.append(client.id)
            history = self.gradient_buffer.setdefault(client.id, [])
            history.append(gradient.copy())
            del history[:-self.buffer_size]

        if self.grad_dim is None:
            self.grad_dim = sum(value.numel() for value in global_state_dict.values())
        self._head_split_idx = head_split_idx if head_split_idx is not None else self.grad_dim
        matrix = np.full((len(clients), self.grad_dim), np.nan, dtype=np.float32)
        for row, gradient in enumerate(gradients):
            if gradient is not None:
                matrix[row] = gradient
        return matrix, straggler_ids, normal_ids

    def _lrmc_single_block(self, block_matrix):
        """对单个梯度块执行LRMC补全"""
        matrix = np.asarray(block_matrix, dtype=np.float32)
        if matrix.size == 0 or np.isnan(matrix).all():
            return np.zeros_like(matrix)
        missing = np.isnan(matrix)
        if not missing.any():
            return matrix.copy()
        observed_rows = np.flatnonzero(~np.isnan(matrix).all(axis=1))
        if len(observed_rows) < 2:
            filled = np.nan_to_num(matrix, nan=0.0)
            if len(observed_rows) == 1:
                filled[np.isnan(matrix).all(axis=1)] = filled[observed_rows[0]]
            return filled

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            column_mean = np.nanmean(matrix, axis=0)
        filled = np.where(missing, np.nan_to_num(column_mean, nan=0.0), matrix)
        # 用历史缓冲池均值作为时序先验
        for row in np.flatnonzero(np.isnan(matrix).all(axis=1)):
            if row < len(self._row_client_ids):
                history = self.gradient_buffer.get(self._row_client_ids[row])
                if history:
                    # 只取对应block的历史梯度
                    block_size = matrix.shape[1]
                    hist_block = [h[:block_size] if len(h) >= block_size else h for h in history]
                    filled[row] = np.mean(hist_block, axis=0)
        max_iter = self.config.get("lrmc_max_iter", 30)
        tolerance = self.config.get("lrmc_tol", 1e-4)
        energy_ratio = min(max(self.config.get("lrmc_rank_ratio", 0.9), 0.0), 1.0)

        # 自适应秩选择：基于特征值衰减率动态调整
        # 如果特征值衰减快，说明低秩结构明显，可以用更低的秩
        K = matrix.shape[0]
        if K > 1:
            try:
                temp_eigenvalues = np.linalg.eigvalsh(filled @ filled.T)
                temp_eigenvalues = np.sort(np.maximum(temp_eigenvalues, 0.0))[::-1]
                if temp_eigenvalues[1] > 0:
                    decay_rate = temp_eigenvalues[0] / temp_eigenvalues[1]
                    if decay_rate > 10:  # 低秩结构明显
                        energy_ratio = min(energy_ratio, 0.7)
            except np.linalg.LinAlgError:
                pass

        for _ in range(max_iter):
            previous = filled.copy()
            try:
                eigenvalues, u = np.linalg.eigh(filled @ filled.T)
            except np.linalg.LinAlgError:
                break
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            u = u[:, order]
            energy = np.cumsum(eigenvalues)
            total = energy[-1] if len(energy) else 0.0
            rank = 1 if total <= 0 else int(np.searchsorted(energy, total * energy_ratio) + 1)
            reconstructed = u[:, :rank] @ (u[:, :rank].T @ filled)
            filled[missing] = reconstructed[missing]
            relative_change = np.linalg.norm(filled - previous) / (np.linalg.norm(previous) + 1e-10)
            if relative_change < tolerance:
                break
        return np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)

    def lrmc_complete(self, grad_matrix):
        """对backbone部分做LRMC补全，分类头部分用列均值填充"""
        matrix = np.asarray(grad_matrix, dtype=np.float32)
        if matrix.size == 0 or np.isnan(matrix).all():
            return np.zeros_like(matrix)
        if not np.isnan(matrix).any():
            return matrix.copy()

        split = self._head_split_idx if self._head_split_idx is not None else matrix.shape[1]

        # Backbone部分：LRMC低秩补全
        backbone_block = matrix[:, :split]
        completed_backbone = self._lrmc_single_block(backbone_block)

        # 分类头部分：列均值填充（不做低秩投影）
        head_block = matrix[:, split:]
        if head_block.shape[1] > 0 and np.isnan(head_block).any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                col_mean = np.nanmean(head_block, axis=0)
            col_mean = np.nan_to_num(col_mean, nan=0.0)
            head_missing = np.isnan(head_block)
            completed_head = np.where(head_missing, col_mean, head_block)
        else:
            completed_head = head_block

        return np.concatenate([completed_backbone, completed_head], axis=1)

    def get_compensated_gradient(self, client_id, lrmc_gradient):
        history = self.gradient_buffer.get(client_id)
        if not history:
            return lrmc_gradient
        # 按轮次衰减加权：越近的梯度权重越大
        n_history = len(history)
        decay_factor = 0.9  # 每过一轮权重衰减10%
        decay_weights = np.array([decay_factor ** (n_history - 1 - i) for i in range(n_history)])
        decay_weights /= decay_weights.sum()
        historical = np.average(history, axis=0, weights=decay_weights)
        lrmc_norm = np.linalg.norm(lrmc_gradient)
        history_norm = np.linalg.norm(historical)
        if lrmc_norm > 0 and history_norm > 0:
            cosine = float(np.dot(lrmc_gradient, historical) / (lrmc_norm * history_norm + 1e-12))
            confidence = max(0.0, cosine)
            lrmc_weight = self.compensation_weight * confidence
            history_weight = 1.0 - lrmc_weight
            result = lrmc_weight * lrmc_gradient + history_weight * historical
            max_norm = 1.5 * history_norm
            result_norm = np.linalg.norm(result)
            if result_norm > max_norm:
                result = result * (max_norm / (result_norm + 1e-12))
            return result
        return historical

    def intra_cluster_aggregate(self, grad_matrix, straggler_ids, clients):
        matrix = np.asarray(grad_matrix).copy()
        if matrix.size == 0:
            return np.zeros(self.grad_dim or 0, dtype=np.float32)
        stragglers = set(straggler_ids)
        for row, client in enumerate(clients):
            if client.id in stragglers:
                matrix[row] = self.get_compensated_gradient(client.id, matrix[row])
        weights = np.asarray([client.num_train_samples for client in clients], dtype=np.float64)
        return np.nan_to_num(np.average(matrix, axis=0, weights=weights), nan=0.0)

    def ota_to_lbs(self, aggregated_grad):
        gradient = np.asarray(aggregated_grad)
        snr_db = self.config.get("ota_snr_db", 0.0)
        if snr_db <= 0 or gradient.size == 0:
            return gradient
        signal_power = np.mean(gradient ** 2)
        if signal_power <= 0:
            return gradient
        noise_std = np.sqrt(signal_power / (10 ** (snr_db / 10)))
        return gradient + np.random.normal(0, noise_std, gradient.shape)
