import numpy as np
import torch
import torch.nn.functional as F


class CloudServer:
    """多模态云端全局服务器 (LBS)"""

    def __init__(self, config):
        self.config = config

    def broadcast_model(self, global_state_dict):
        return global_state_dict

    def apply_update(self, global_state_dict, global_update_1d, device):
        if global_update_1d is None:
            return
        update = np.asarray(global_update_1d).reshape(-1)
        expected = sum(value.numel() for value in global_state_dict.values())
        if update.size != expected or not np.isfinite(update).all():
            raise ValueError(f"Invalid update length: got {update.size}, expected {expected}")

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

    def evaluate(self, global_state, test_loader, model, device):
        model.load_state_dict(global_state)
        model.eval()
        correct = 0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for (images, texts), labels in test_loader:
                images = images.to(device, non_blocking=True)
                texts = texts.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model((images, texts))
                total_loss += F.cross_entropy(logits, labels, reduction='sum').item()
                correct += logits.argmax(dim=1).eq(labels).sum().item()
                total_samples += labels.size(0)

        acc = 100.0 * correct / max(total_samples, 1)
        loss = total_loss / max(total_samples, 1)
        return acc, loss