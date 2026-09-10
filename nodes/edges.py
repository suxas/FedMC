import warnings
import numpy as np

class EdgeSBS:
    """边缘小基站 (SBS) - 负责模态解耦 LRMC 重构与簇内聚合"""
    def __init__(self, sbs_id, config):
        self.id = sbs_id
        self.config = config
        self.buffer_size = config.get("buffer_size", 6)
        self.compensation_weight = config.get("compensation_weight", 0.75)
        self.gradient_buffer = {}
        self.grad_dim = None
        self._row_client_ids = []
        self._vis_end_idx = None
        self._head_split_idx = None
        self._current_round = 0

    def _inspect_layer_indices(self, global_state_dict):
        v_size, t_size, c_size = 0, 0, 0
        for name, tensor in global_state_dict.items():
            if not tensor.is_floating_point():
                continue
            n = tensor.numel()
            if 'visual' in name:
                v_size += n
            elif 'text' in name:
                t_size += n
            else:
                c_size += n
        return v_size, v_size + t_size, v_size + t_size + c_size

    def collect_gradients(self, clients, global_state_dict, round_idx=0):
        self._current_round = round_idx
        gradients, straggler_ids, normal_ids = [], [], []
        self._row_client_ids = [c.id for c in clients]

        if self._vis_end_idx is None:
            v_end, t_end, tot = self._inspect_layer_indices(global_state_dict)
            self._vis_end_idx = v_end
            self._head_split_idx = t_end
            self.grad_dim = tot

        times = [c.simulate_communication_time(round_idx)[1] for c in clients]
        straggler_ratio = min(max(self.config.get("straggler_ratio", 0.5), 0.0), 1.0)
        k = max(1, int(len(clients) * (1.0 - straggler_ratio))) - 1
        t_win = sorted(times)[k]

        for client, t in zip(clients, times):
            if t > t_win:
                gradients.append(None)
                straggler_ids.append(client.id)
                continue

            grad, _ = client.local_train(global_state_dict)
            grad = np.asarray(grad).reshape(-1)
            gradients.append(grad)
            normal_ids.append(client.id)

            history = self.gradient_buffer.setdefault(client.id, [])
            history.append(grad.copy())
            del history[:-self.buffer_size]

        matrix = np.full((len(clients), self.grad_dim), np.nan, dtype=np.float32)
        for row, grad in enumerate(gradients):
            if grad is not None:
                matrix[row] = grad

        # 异步更新机制：慢节点后台完成计算并缓存，供后续重构使用
        for sid in straggler_ids:
            target_client = next(c for c in clients if c.id == sid)
            async_grad, _ = target_client.local_train(global_state_dict)
            h = self.gradient_buffer.setdefault(sid, [])
            h.append(np.asarray(async_grad).reshape(-1).copy())
            del h[:-self.buffer_size]

        return matrix, straggler_ids, normal_ids

    def _lrmc_visual_block(self, visual_mat, round_idx=None):
        """对视觉主干稠密卷积核执行严密的 SVD 低秩补全 (集成退火截断)"""
        missing = np.isnan(visual_mat)
        if not missing.any():
            return visual_mat.copy()

        observed_rows = np.flatnonzero(~missing.all(axis=1))
        if len(observed_rows) == 0:
            return np.zeros_like(visual_mat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_mean = np.nanmean(visual_mat, axis=0)
        col_mean = np.nan_to_num(col_mean, nan=0.0)
        filled = np.where(missing, col_mean, visual_mat)

        # 优先使用掉队节点的历史视觉参数作为补全初始估计
        for r_idx, cid in enumerate(self._row_client_ids):
            if missing[r_idx].all() and cid in self.gradient_buffer and len(self.gradient_buffer[cid]) > 0:
                filled[r_idx] = self.gradient_buffer[cid][-1][:visual_mat.shape[1]]

        max_iter = self.config.get("lrmc_max_iter", 30)
        tol = self.config.get("lrmc_tol", 1e-4)

        # 自适应平滑退火能量截断比: 前期过滤大尺度噪声，后期自适应抬升保留细粒度特征
        if round_idx is None:
            round_idx = self._current_round
        total_rounds = max(1, self.config.get("global_rounds", 300))
        rank_min = self.config.get("lrmc_rank_ratio_min", self.config.get("lrmc_rank_ratio", 0.80))
        rank_max = self.config.get("lrmc_rank_ratio_max", 0.95)
        progress = min(max(round_idx / float(total_rounds), 0.0), 1.0)
        energy_ratio = rank_min + (rank_max - rank_min) * progress

        for _ in range(max_iter):
            prev = filled.copy()
            gram = filled @ filled.T
            try:
                eigvals, u = np.linalg.eigh(gram)
            except np.linalg.LinAlgError:
                break

            order = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[order], 0.0)
            u = u[:, order]

            energy = np.cumsum(eigvals)
            tot = energy[-1] if len(energy) else 0.0
            rank = 1 if tot <= 0 else int(np.searchsorted(energy, tot * energy_ratio) + 1)
            rank = max(1, min(rank, len(observed_rows)))

            sub_u = u[:, :rank]
            reconstructed = sub_u @ (sub_u.T @ filled)
            filled[missing] = reconstructed[missing]

            if np.linalg.norm(filled - prev) / (np.linalg.norm(prev) + 1e-10) < tol:
                break

        return np.nan_to_num(filled, nan=0.0)

    def lrmc_complete(self, grad_matrix, round_idx=None):
        matrix = np.asarray(grad_matrix, dtype=np.float32)
        if matrix.size == 0 or not np.isnan(matrix).any():
            return np.nan_to_num(matrix, nan=0.0)

        if round_idx is None:
            round_idx = self._current_round

        v_end = self._vis_end_idx

        # 1. 视觉卷积块: 执行带退火机制的高精度 LRMC
        v_block = matrix[:, :v_end]
        completed_visual = self._lrmc_visual_block(v_block, round_idx=round_idx)

        # 2. 文本编码器与分类头: 弱相关与类别特异性，改用历史动量保护
        rest_block = matrix[:, v_end:].copy()
        rest_missing = np.isnan(rest_block)
        if rest_missing.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_rest = np.nanmean(rest_block, axis=0)
            mean_rest = np.nan_to_num(mean_rest, nan=0.0)
            rest_block = np.where(rest_missing, mean_rest, rest_block)

            for r_idx, cid in enumerate(self._row_client_ids):
                if rest_missing[r_idx].any() and cid in self.gradient_buffer and len(self.gradient_buffer[cid]) > 0:
                    rest_block[r_idx] = self.gradient_buffer[cid][-1][v_end:]

        return np.concatenate([completed_visual, rest_block], axis=1)

    def get_compensated_gradient(self, client_id, lrmc_gradient):
        history = self.gradient_buffer.get(client_id)
        if not history:
            return lrmc_gradient

        n_hist = len(history)
        decay = np.array([0.85 ** (n_hist - 1 - i) for i in range(n_hist)])
        decay /= decay.sum()
        historical = np.average(history, axis=0, weights=decay)

        norm_l = np.linalg.norm(lrmc_gradient)
        norm_h = np.linalg.norm(historical)
        if norm_l > 0 and norm_h > 0:
            cosine = float(np.dot(lrmc_gradient, historical) / (norm_l * norm_h + 1e-12))
            confidence = max(0.0, cosine)
            alpha = self.compensation_weight * (0.5 + 0.5 * confidence)
            return alpha * lrmc_gradient + (1.0 - alpha) * historical
        return historical if norm_h > 0 else lrmc_gradient

    def intra_cluster_aggregate(self, grad_matrix, straggler_ids, clients):
        matrix = np.asarray(grad_matrix).copy()
        if matrix.size == 0:
            return np.zeros(self.grad_dim or 0, dtype=np.float32), 0

        stragglers = set(straggler_ids)
        for row, client in enumerate(clients):
            if client.id in stragglers:
                matrix[row] = self.get_compensated_gradient(client.id, matrix[row])

        sample_counts = np.array([c.num_train_samples for c in clients], dtype=np.float64)
        cluster_sum = np.sum(matrix * sample_counts[:, None], axis=0)
        return np.nan_to_num(cluster_sum, nan=0.0), int(np.sum(sample_counts))

    def ota_to_lbs(self, aggregated_grad):
        gradient = np.asarray(aggregated_grad)
        if not self.config.get("enable_ota_noise", False) or gradient.size == 0:
            return gradient

        snr_db = self.config.get("ota_snr_db", 20.0)
        signal_power = np.mean(gradient ** 2)
        if signal_power <= 0:
            return gradient
        noise_std = np.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
        return gradient + np.random.normal(0, noise_std, gradient.shape)