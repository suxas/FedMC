import numpy as np
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import SharedBackbone
from utils.dataset import get_dataloaders
from main import ota_aggregate_gradients


def evaluate_metrics_with_loss(clients, global_backbone_state, device):
    cmc5_scores, r1_scores, map_scores, losses = [], [], [], []
    criterion = torch.nn.CrossEntropyLoss()

    for client in clients:
        client.set_global_features(global_backbone_state)
        client.encoder.eval()
        client.shared_backbone.eval()
        client.classifier.eval()

        correct1, correct5, total, running_loss = 0, 0, 0, 0.0
        client_preds, client_labels = [], []

        with torch.no_grad():
            for img_1, img_2, labels in client.test_loader:
                if len(labels) == 0: continue
                labels = labels.to(device)
                inputs = img_1.to(device) if client.modal_type == "RGB" else img_2.to(device)

                outputs = client.classifier(client.shared_backbone(client.encoder(inputs)))
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                _, top5_preds = outputs.topk(5, 1, True, True)
                top5_preds = top5_preds.t()
                correct = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))

                correct1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                correct5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                total += labels.size(0)

                _, predicted = outputs.max(1)
                client_preds.extend(predicted.cpu().numpy())
                client_labels.extend(labels.cpu().numpy())

        if total > 0:
            losses.append(running_loss / total)
            r1_scores.append(100. * correct1 / total)
            cmc5_scores.append(100. * correct5 / total)

            preds_arr = np.array(client_preds)
            labels_arr = np.array(client_labels)
            ap_list = [1.0 if p == l else 0.0 for p, l in zip(preds_arr, labels_arr)]
            map_scores.append(np.mean(ap_list) * 100 if len(ap_list) > 0 else 0.0)

    return np.mean(cmc5_scores), np.mean(r1_scores), np.mean(map_scores), np.mean(losses)


def run_fedavg(config):
    train_loaders, test_loaders, num_classes = get_dataloaders(config)
    global_model = SharedBackbone(config['dataset']).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = []
    for i in range(config['num_sbs'] * config['clients_per_sbs']):
        modal = "RGB" if (i % 2 == 0) else "IR"
        clients.append(ClientUE(i, config, train_loaders[i], test_loaders[i], num_classes, modal))

    cmc5_history, r1_history, map_history, loss_history = [], [], [], []
    pbar = tqdm(range(config['global_rounds']), colour='red', desc="[OTA FedAvg]")

    for round_idx in pbar:
        edge_grads = []
        for sbs_id in range(config['num_sbs']):
            cluster_clients = clients[sbs_id * config['clients_per_sbs']: (sbs_id + 1) * config['clients_per_sbs']]

            raw_grads = [c.local_train(global_state_dict) for c in cluster_clients]
            ota_grad = ota_aggregate_gradients(raw_grads, config)

            if np.random.rand() >= config['ota_drop_rate']:
                edge_grads.append(ota_grad)

        if edge_grads:
            global_update_1d = ota_aggregate_gradients(edge_grads, config)
            idx = 0
            for name, param in global_state_dict.items():
                size = param.numel()
                # 【FedBN 服务器端】：不更新全局 BN
                if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                    idx += size
                    continue

                update_tensor = torch.tensor(global_update_1d[idx:idx + size]).reshape(param.shape).to(config['device'])
                updated_val = global_state_dict[name].float() - update_tensor.float()

                if not global_state_dict[name].is_floating_point():
                    global_state_dict[name].copy_(updated_val.round().to(param.dtype))
                else:
                    global_state_dict[name].copy_(updated_val.to(param.dtype))

                idx += size

        mean_cmc5, mean_r1, mean_map, mean_loss = evaluate_metrics_with_loss(clients, global_state_dict,
                                                                             config['device'])

        cmc5_history.append(mean_cmc5)
        r1_history.append(mean_r1)
        map_history.append(mean_map)
        loss_history.append(mean_loss)

        pbar.set_postfix(CMC5=f"{mean_cmc5:.2f}%", R1=f"{mean_r1:.2f}%", Loss=f"{mean_loss:.4f}")

    return {"cmc5": cmc5_history, "r1": r1_history, "map": map_history, "loss": loss_history}