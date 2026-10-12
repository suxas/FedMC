import warnings
import numpy as np

class EdgeSBS:
    """边缘小基站 (SBS) - 集成模态解耦残差 LRMC、模长校准与抗污染真余弦门控重构"""
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
        self._last_global_state = None
        self._current_round_gradients = {}

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

    def collect_gradients(self, clients, global_state_dict, round_idx=0, tau_override=None):
        self._current_round = round_idx
        self._last_global_state = global_state_dict
        self._current_round_gradients = {}
        gradients, straggler_ids, normal_ids = [], [], []
        self._row_client_ids = [c.id for c in clients]

        if self._vis_end_idx is None:
            v_end, t_end, tot = self._inspect_layer_indices(global_state_dict)
            self._vis_end_idx = v_end
            self._head_split_idx = t_end
            self.grad_dim = tot

        times = [c.simulate_communication_time(round_idx)[1] for c in clients]
        straggler_ratio = min(max(self.config.get("straggler_ratio", 0.5), 0.0), 1.0)
        if tau_override is not None:
            t_win = tau_override
        else:
            k = max(1, int(len(clients) * (1.0 - straggler_ratio))) - 1
            t_win = sorted(times)[k]

        delta_matrix = np.full((len(clients), self.grad_dim), np.nan, dtype=np.float32)

        for row, (client, t) in enumerate(zip(clients, times)):
            if t > t_win:
                gradients.append(None)
                straggler_ids.append(client.id)
                continue

            grad, _ = client.local_train(global_state_dict)
            grad = np.asarray(grad, dtype=np.float32).reshape(-1)
            gradients.append(grad)
            normal_ids.append(client.id)
            self._current_round_gradients[client.id] = grad

            history = self.gradient_buffer.get(client.id)
            if history and len(history) > 0:
                delta_matrix[row] = grad - history[-1]
            else:
                delta_matrix[row] = grad.copy()

        return delta_matrix, straggler_ids, normal_ids

    def _lrmc_visual_block(self, visual_delta_mat, round_idx=None):
        missing = np.isnan(visual_delta_mat)
        if not missing.any():
            return visual_delta_mat.copy()

        observed_rows = np.flatnonzero(~missing.all(axis=1))
        if len(observed_rows) == 0:
            return np.zeros_like(visual_delta_mat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_mean = np.nanmean(visual_delta_mat, axis=0)
        col_mean = np.nan_to_num(col_mean, nan=0.0)
        filled = np.where(missing, col_mean, visual_delta_mat)

        max_iter = self.config.get("lrmc_max_iter", 30)
        tol = self.config.get("lrmc_tol", 1e-4)

        if round_idx is None:
            round_idx = self._current_round
        total_rounds = max(1, self.config.get("global_rounds", 500))
        rank_min = self.config.get("lrmc_rank_ratio_min", 0.80)
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

    def lrmc_complete(self, delta_matrix, round_idx=None):
        delta_mat = np.asarray(delta_matrix, dtype=np.float32)
        if delta_mat.size == 0:
            return np.zeros((len(self._row_client_ids), self.grad_dim or 0), dtype=np.float32)

        if round_idx is None:
            round_idx = self._current_round

        v_end = self._vis_end_idx

        # 1. 视觉主干残差增量执行退火 LRMC 补全
        v_delta = delta_mat[:, :v_end]
        completed_v_delta = self._lrmc_visual_block(v_delta, round_idx=round_idx)

        # 2. 文本与分类头：掉队行严格置零以锁定私有类别梯度
        rest_delta = np.zeros((delta_mat.shape[0], delta_mat.shape[1] - v_end), dtype=np.float32)
        for r_idx in range(delta_mat.shape[0]):
            if not np.isnan(delta_mat[r_idx, v_end:]).any():
                rest_delta[r_idx] = delta_mat[r_idx, v_end:]

        completed_delta = np.concatenate([completed_v_delta, rest_delta], axis=1)

        # 3. 模长自适应校准 (Energy Alignment)
        observed_mask = ~np.isnan(delta_mat).any(axis=1)
        if observed_mask.any():
            active_norms = np.linalg.norm(completed_delta[observed_mask, :v_end], axis=1)
            mean_target_norm = float(np.mean(active_norms))
        else:
            mean_target_norm = 1.0

        full_grad_matrix = np.empty((len(self._row_client_ids), self.grad_dim), dtype=np.float32)

        for row, cid in enumerate(self._row_client_ids):
            if cid in self._current_round_gradients:
                full_grad_matrix[row] = self._current_round_gradients[cid]
            else:
                cur_recon_norm = np.linalg.norm(completed_delta[row, :v_end])
                if cur_recon_norm > 1e-8 and mean_target_norm > 1e-8:
                    scale_factor = min(max(mean_target_norm / cur_recon_norm, 0.8), 1.6)
                    completed_delta[row, :v_end] *= scale_factor

                history = self.gradient_buffer.get(cid)
                if history and len(history) > 0:
                    hist_grad = history[-1]
                    recon_grad = hist_grad + completed_delta[row]

                    norm_r = np.linalg.norm(recon_grad)
                    norm_h = np.linalg.norm(hist_grad)
                    if norm_r > 0 and norm_h > 0:
                        cos = float(np.dot(recon_grad, hist_grad) / (norm_r * norm_h + 1e-12))
                        cos_c = max(-1.0, min(1.0, cos))
                        alpha = self.compensation_weight * (0.5 + 0.5 * max(0.0, cos_c))
                        full_grad_matrix[row] = alpha * recon_grad + (1.0 - alpha) * hist_grad
                    else:
                        full_grad_matrix[row] = recon_grad
                else:
                    full_grad_matrix[row] = completed_delta[row]

        return np.nan_to_num(full_grad_matrix, nan=0.0)

    def intra_cluster_aggregate(self, grad_matrix, straggler_ids, clients):
        matrix = np.asarray(grad_matrix).copy()
        if matrix.size == 0:
            return np.zeros(self.grad_dim or 0, dtype=np.float32), 0

        sample_counts = np.array([c.num_train_samples for c in clients], dtype=np.float64)
        cluster_sum = np.sum(matrix * sample_counts[:, None], axis=0)

        stragglers = set(straggler_ids)
        for client in clients:
            if client.id in stragglers:
                async_grad, _ = client.local_train(self._last_global_state)
                h = self.gradient_buffer.setdefault(client.id, [])
                h.append(np.asarray(async_grad, dtype=np.float32).reshape(-1).copy())
                del h[:-self.buffer_size]
            else:
                if client.id in self._current_round_gradients:
                    h = self.gradient_buffer.setdefault(client.id, [])
                    h.append(self._current_round_gradients[client.id].copy())
                    del h[:-self.buffer_size]

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