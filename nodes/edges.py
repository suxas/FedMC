import warnings
import numpy as np


class EdgeSBS:
    """边缘基站 SBS：多模态骨干联合矩阵补全、动量融合与簇内聚合"""

    def __init__(self, sbs_id, config):
        self.id = sbs_id
        self.config = config
        self.buffer_size = config.get("buffer_size", 5)
        self.compensation_weight = config.get("compensation_weight", 0.8)
        self.gradient_buffer = {}
        self.grad_dim = None
        self._row_client_ids = []
        self._head_split_idx = None

    def collect_gradients(self, clients, global_state_dict, round_idx=0):
        gradients, straggler_ids, normal_ids = [], [], []
        self._row_client_ids = [client.id for client in clients]
        head_split_idx = None

        times = []
        for client in clients:
            _, total_time = client.simulate_communication_time(round_idx)
            times.append(total_time)

        straggler_ratio = min(max(self.config.get("straggler_ratio", 0.5), 0.0), 1.0)
        k = max(1, int(len(clients) * (1.0 - straggler_ratio))) - 1
        t_win = sorted(times)[k]

        for client, t in zip(clients, times):
            if t > t_win:
                gradients.append(None)
                straggler_ids.append(client.id)
                continue

            gradient, split_idx = client.local_train(global_state_dict)
            gradient = np.asarray(gradient).reshape(-1)
            if head_split_idx is None:
                head_split_idx = split_idx
            if self.grad_dim is None:
                self.grad_dim = gradient.size

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
        matrix = np.asarray(block_matrix, dtype=np.float32)
        if matrix.size == 0 or np.isnan(matrix).all():
            return np.zeros_like(matrix)
        missing = np.isnan(matrix)
        if not missing.any():
            return matrix.copy()

        observed_rows = np.flatnonzero(~np.isnan(matrix).all(axis=1))
        if len(observed_rows) == 0:
            return np.zeros_like(matrix)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            column_mean = np.nanmean(matrix, axis=0)
        filled = np.where(missing, np.nan_to_num(column_mean, nan=0.0), matrix)

        for row in np.flatnonzero(np.isnan(matrix).all(axis=1)):
            if row < len(self._row_client_ids):
                history = self.gradient_buffer.get(self._row_client_ids[row])
                if history:
                    block_size = matrix.shape[1]
                    hist_block = [h[:block_size] if len(h) >= block_size else h for h in history]
                    filled[row] = np.mean(hist_block, axis=0)

        max_iter = self.config.get("lrmc_max_iter", 30)
        tolerance = self.config.get("lrmc_tol", 1e-4)
        energy_ratio = min(max(self.config.get("lrmc_rank_ratio", 0.8), 0.0), 1.0)

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

        return np.nan_to_num(filled, nan=0.0)

    def lrmc_complete(self, grad_matrix):
        matrix = np.asarray(grad_matrix, dtype=np.float32)
        if matrix.size == 0 or not np.isnan(matrix).any():
            return np.nan_to_num(matrix, nan=0.0)

        split = self._head_split_idx if self._head_split_idx is not None else matrix.shape[1]

        backbone_block = matrix[:, :split]
        completed_backbone = self._lrmc_single_block(backbone_block)

        head_block = matrix[:, split:]
        if head_block.shape[1] > 0 and np.isnan(head_block).any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                col_mean = np.nanmean(head_block, axis=0)
            col_mean = np.nan_to_num(col_mean, nan=0.0)
            completed_head = np.where(np.isnan(head_block), col_mean, head_block)
            for row in np.flatnonzero(np.isnan(head_block).all(axis=1)):
                if row < len(self._row_client_ids):
                    history = self.gradient_buffer.get(self._row_client_ids[row])
                    if history:
                        hist_head = [h[split:] for h in history if len(h) > split]
                        if hist_head:
                            completed_head[row] = np.mean(hist_head, axis=0)
        else:
            completed_head = head_block

        return np.concatenate([completed_backbone, completed_head], axis=1)

    def get_compensated_gradient(self, client_id, lrmc_gradient):
        history = self.gradient_buffer.get(client_id)
        if not history:
            return lrmc_gradient

        n_history = len(history)
        decay_factor = 0.9
        decay_weights = np.array([decay_factor ** (n_history - 1 - i) for i in range(n_history)])
        decay_weights /= decay_weights.sum()
        historical = np.average(history, axis=0, weights=decay_weights)

        lrmc_norm = np.linalg.norm(lrmc_gradient)
        history_norm = np.linalg.norm(historical)

        if lrmc_norm > 0 and history_norm > 0:
            cosine = float(np.dot(lrmc_gradient, historical) / (lrmc_norm * history_norm + 1e-12))
            confidence = max(0.0, cosine)
            alpha = self.compensation_weight * (0.5 + 0.5 * confidence)
            result = alpha * lrmc_gradient + (1.0 - alpha) * historical
            return result
        return historical if history_norm > 0 else lrmc_gradient

    def intra_cluster_aggregate(self, grad_matrix, straggler_ids, clients):
        matrix = np.asarray(grad_matrix).copy()
        if matrix.size == 0:
            return np.zeros(self.grad_dim or 0, dtype=np.float32), 0

        stragglers = set(straggler_ids)
        for row, client in enumerate(clients):
            if client.id in stragglers:
                matrix[row] = self.get_compensated_gradient(client.id, matrix[row])

        sample_counts = np.array([c.num_train_samples for c in clients], dtype=np.float64)
        cluster_sum_grad = np.sum(matrix * sample_counts[:, None], axis=0)
        total_samples = int(np.sum(sample_counts))
        return np.nan_to_num(cluster_sum_grad, nan=0.0), total_samples

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