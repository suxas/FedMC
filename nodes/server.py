import numpy as np
import torch
import torch.nn.functional as F


class CloudServer:
    """Cloud aggregation and classification evaluation."""

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
            raise ValueError(f"Invalid global update length/values: got {update.size}, expected {expected}")

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

    def _classification_loss_and_acc(self, clients, global_state, device):
        losses, accuracies = [], []
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        for client in clients:
            client.set_global_features(global_state)
            client.set_eval_mode()
            total_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in client.test_loader:
                    if len(labels) == 0:
                        continue
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    logits = client.model(images)
                    total_loss += criterion(logits, labels).item()
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.numel()
            if total:
                losses.append(total_loss / total)
                accuracies.append(correct / total * 100.0)
        return float(np.mean(losses)) if losses else 0.0, float(np.mean(accuracies)) if accuracies else 0.0

    def evaluate(self, clients, global_state, device):
        if not clients:
            return 0.0, 0.0, 0.0
        loss, acc = self._classification_loss_and_acc(clients, global_state, device)
        return acc, acc, loss
