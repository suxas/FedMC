import numpy as np
import torch


class CloudServer:
    """大基站 (LBS) 云端服务器：全局模型管理、聚合、评估"""

    def __init__(self, config):
        self.config = config

    def broadcast_model(self, global_state_dict):
        """广播全局模型到所有 SBS（返回 state_dict 引用）"""
        return global_state_dict

    def global_aggregate(self, edge_gradients):
        """对各 SBS 的梯度做全局均值聚合

        Args:
            edge_gradients: list of np.array，每个 SBS 聚合后的梯度

        Returns:
            global_update: 全局梯度更新向量
        """
        valid_grads = [g for g in edge_gradients if g is not None and np.sum(np.abs(g)) > 0]
        if not valid_grads:
            return None
        return np.mean(valid_grads, axis=0)

    def apply_update(self, global_state_dict, global_update_1d, device):
        """将全局梯度更新应用到模型权重

        Args:
            global_state_dict: 当前全局模型 state_dict
            global_update_1d: 一维梯度更新向量
            device: 计算设备
        """
        if global_update_1d is None:
            return

        idx = 0
        for name, param in global_state_dict.items():
            size = param.numel()
            # FedBN: 跳过 BN 统计量
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                idx += size
                continue

            update_tensor = torch.tensor(global_update_1d[idx:idx + size]).reshape(param.shape).to(device)
            updated_val = global_state_dict[name].float() - update_tensor.float()

            if not global_state_dict[name].is_floating_point():
                global_state_dict[name].copy_(updated_val.round().to(param.dtype))
            else:
                global_state_dict[name].copy_(updated_val.to(param.dtype))

            idx += size

    def evaluate(self, clients, global_backbone_state, device):
        """使用全局模型在所有客户端上评估

        Returns:
            r1: 平均 Rank-1 准确率 (%)
            mAP: 平均 mAP (%)
            loss: 平均测试损失
        """
        r1_scores, map_scores, losses = [], [], []
        criterion = torch.nn.CrossEntropyLoss()

        for client in clients:
            client.set_global_features(global_backbone_state)

            # 设置评估模式
            if client.encoder_rgb is not None:
                client.encoder_rgb.eval()
            if client.encoder_ir is not None and client.encoder_ir is not client.encoder_rgb:
                client.encoder_ir.eval()
            if client.fusion is not None:
                client.fusion.eval()
            client.shared_backbone.eval()
            client.classifier.eval()

            correct1, total, running_loss = 0, 0, 0.0
            client_preds, client_labels = [], []

            with torch.no_grad():
                for img_1, img_2, labels in client.test_loader:
                    if len(labels) == 0:
                        continue
                    labels = labels.to(device)
                    img_rgb = img_1.to(device)
                    img_ir = img_2.to(device)

                    outputs = client._forward(img_rgb, img_ir)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * labels.size(0)

                    _, predicted = outputs.max(1)
                    correct1 += predicted.eq(labels).sum().item()
                    total += labels.size(0)

                    client_preds.extend(predicted.cpu().numpy())
                    client_labels.extend(labels.cpu().numpy())

            if total > 0:
                losses.append(running_loss / total)
                r1_scores.append(100.0 * correct1 / total)

                # mAP 计算（简化版：逐样本精确匹配）
                preds_arr = np.array(client_preds)
                labels_arr = np.array(client_labels)
                ap_list = [1.0 if p == l else 0.0 for p, l in zip(preds_arr, labels_arr)]
                map_scores.append(np.mean(ap_list) * 100 if len(ap_list) > 0 else 0.0)

        mean_r1 = np.mean(r1_scores) if r1_scores else 0.0
        mean_map = np.mean(map_scores) if map_scores else 0.0
        mean_loss = np.mean(losses) if losses else 0.0

        return mean_r1, mean_map, mean_loss
