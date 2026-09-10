import numpy as np
import torch
import torch.nn.functional as F

class EvalResult(dict):
    def __iter__(self):
        return iter((self.get('top1', self.get('acc', 0.0)), self.get('loss', 0.0)))

class CloudServer:
    """云端中央服务器 (LBS)"""
    def __init__(self, config):
        self.config = config
        # FedOpt (FedAdam) 优化器状态
        self.m = None
        self.v = None
        self.adam_t = 0

    def broadcast_model(self, global_state_dict):
        return global_state_dict

    def apply_update(self, global_state_dict, global_update_1d, device):
        """标准 FedAvg 加权伪梯度应用"""
        if global_update_1d is None:
            return
        update = np.asarray(global_update_1d).reshape(-1)
        expected = sum(value.numel() for value in global_state_dict.values())
        if update.size != expected or not np.isfinite(update).all():
            raise ValueError(f"Invalid update size: got {update.size}, expected {expected}")

        offset = 0
        with torch.no_grad():
            for name, value in global_state_dict.items():
                size = value.numel()
                segment = update[offset:offset + size]
                offset += size
                if not value.is_floating_point():
                    continue
                delta = torch.as_tensor(segment, device=device, dtype=torch.float32).reshape(value.shape)
                value.copy_((value.float() - delta).to(value.dtype))

    def apply_adam_update(self, global_state_dict, pseudo_grad_1d, device):
        """严格遵守 ICLR 2021 原生 FedAdam 服务端自适应更新逻辑"""
        if pseudo_grad_1d is None:
            return
        g = np.asarray(pseudo_grad_1d, dtype=np.float32).reshape(-1)
        expected = sum(value.numel() for value in global_state_dict.values())
        if g.size != expected or not np.isfinite(g).all():
            raise ValueError(f"Invalid pseudo gradient size: got {g.size}, expected {expected}")

        beta1 = self.config.get("server_beta1", 0.9)
        beta2 = self.config.get("server_beta2", 0.99)
        eps = self.config.get("server_eps", 1e-3)
        eta = self.config.get("server_lr", 0.01)

        if self.m is None:
            self.m = np.zeros_like(g)
            self.v = np.zeros_like(g)

        self.adam_t += 1
        # 一阶动量与二阶动量递推
        self.m = beta1 * self.m + (1.0 - beta1) * g
        self.v = beta2 * self.v + (1.0 - beta2) * (g ** 2)

        # 偏差修正
        m_hat = self.m / (1.0 - (beta1 ** self.adam_t))
        v_hat = self.v / (1.0 - (beta2 ** self.adam_t))
        adam_delta = (eta * m_hat) / (np.sqrt(v_hat) + eps)

        offset = 0
        with torch.no_grad():
            for name, value in global_state_dict.items():
                size = value.numel()
                segment = adam_delta[offset:offset + size]
                offset += size
                if not value.is_floating_point():
                    continue
                delta_tensor = torch.as_tensor(segment, device=device, dtype=torch.float32).reshape(value.shape)
                value.copy_((value.float() - delta_tensor).to(value.dtype))

    @staticmethod
    def _compute_macro_f1(preds, targets):
        try:
            from sklearn.metrics import f1_score
            return float(f1_score(targets, preds, average='macro', zero_division=0) * 100.0)
        except Exception:
            classes = np.unique(np.concatenate([preds, targets]))
            f1_list = []
            for c in classes:
                tp = np.sum((preds == c) & (targets == c))
                fp = np.sum((preds == c) & (targets != c))
                fn = np.sum((preds != c) & (targets == c))
                denom = 2 * tp + fp + fn
                f1_list.append((2.0 * tp / denom) if denom > 0 else 0.0)
            return float(np.mean(f1_list) * 100.0) if len(f1_list) > 0 else 0.0

    def evaluate(self, global_state, test_loader, model, device):
        model.load_state_dict(global_state)
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_top1 = 0
        correct_top5 = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for (images, texts), labels in test_loader:
                images = images.to(device, non_blocking=True)
                texts = texts.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model((images, texts), drop_visual=False)
                total_loss += F.cross_entropy(logits, labels, reduction='sum').item()
                batch_size = labels.size(0)
                total_samples += batch_size

                preds = logits.argmax(dim=1)
                correct_top1 += preds.eq(labels).sum().item()

                k = min(5, logits.size(1))
                _, top5_indices = logits.topk(k, dim=1, largest=True, sorted=True)
                correct_top5 += top5_indices.eq(labels.view(-1, 1)).sum().item()

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        num_samples = max(total_samples, 1)
        all_preds = np.concatenate(all_preds) if all_preds else np.array([])
        all_labels = np.concatenate(all_labels) if all_labels else np.array([])

        return EvalResult({
            "loss": float(total_loss / num_samples),
            "top1": float(100.0 * correct_top1 / num_samples),
            "top5": float(100.0 * correct_top5 / num_samples),
            "f1": float(self._compute_macro_f1(all_preds, all_labels)),
            "acc": float(100.0 * correct_top1 / num_samples)
        })