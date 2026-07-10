import numpy as np
import warnings
import torch
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders
from nodes.models import FeatureExtractor


def dynamic_sparsify(grad, sparsity_ratio):
    threshold = np.percentile(np.abs(grad), sparsity_ratio * 100)
    return np.where(np.abs(grad) >= threshold, grad, np.nan)


def low_rank_matrix_completion(mat):
    if mat.size == 0 or np.isnan(mat).all():
        return mat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(mat, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    nan_mask = np.where(np.isnan(mat))

    mat_filled = np.copy(mat)
    mat_filled[nan_mask] = np.take(col_mean, nan_mask[1])
    mat_filled = np.nan_to_num(mat_filled, nan=0.0)

    U, s, Vt = np.linalg.svd(mat_filled, full_matrices=False)
    rank = max(1, int(len(s) * 0.2))
    s[rank:] = 0
    mat_reconstructed = np.dot(U, np.dot(np.diag(s), Vt))

    mat_out = np.copy(mat)
    mat_out[nan_mask] = mat_reconstructed[nan_mask]
    return mat_out


def evaluate_global_model(clients, global_state_dict, test_loader, device):
    accs, losses = [], []
    criterion = torch.nn.CrossEntropyLoss()
    for client in clients:
        client.set_global_features(global_state_dict)
        client.feature_extractor.eval()
        client.classifier.eval()
        correct, total, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = client.feature_extractor(images)
                outputs = client.classifier(features)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accs.append(100. * correct / total)
        losses.append(running_loss / total)
    return np.mean(accs), np.mean(losses)


def run_proposed(config):
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    in_channels = 3 if config['dataset'] == "CIFAR10" else 1
    global_model = FeatureExtractor(in_channels).to(config['device'])
    global_state_dict = global_model.state_dict()
    config['grad_dim'] = sum(p.numel() for p in global_state_dict.values())

    server = CloudServer(config)
    sbs_list = [EdgeSBS(i, config) for i in range(config['num_sbs'])]
    clients = [ClientUE(i, config, train_loaders[i], num_classes) for i in
               range(config['num_sbs'] * config['clients_per_sbs'])]

    accuracy_history, loss_history = [], []
    pbar = tqdm(range(config['global_rounds']), colour='blue', desc="[Proposed LRMC]")

    for round_idx in pbar:
        for c in clients:
            c.refresh_state()

        edge_grads = []
        for sbs in sbs_list:
            start = sbs.id * config['clients_per_sbs']
            end = start + config['clients_per_sbs']
            mat_list_rgb, mat_list_nir = [], []

            for client in clients[start:end]:
                raw_grad = client.local_train(global_state_dict)
                if client.channel_quality >= config['noma_threshold']:
                    sparsity = 0.1 if client.is_straggler_state else 0.5
                    processed_grad = dynamic_sparsify(raw_grad, sparsity)
                else:
                    processed_grad = np.full_like(raw_grad, np.nan)

                if client.modality == 'RGB':
                    mat_list_rgb.append(processed_grad)
                else:
                    mat_list_nir.append(processed_grad)

            grad_matrix_rgb = np.array(mat_list_rgb) if mat_list_rgb else np.empty((0, config['grad_dim']))
            grad_matrix_nir = np.array(mat_list_nir) if mat_list_nir else np.empty((0, config['grad_dim']))

            completed_rgb = low_rank_matrix_completion(grad_matrix_rgb)
            completed_nir = low_rank_matrix_completion(grad_matrix_nir)

            valid_matrices = []
            if completed_rgb.size > 0: valid_matrices.append(completed_rgb)
            if completed_nir.size > 0: valid_matrices.append(completed_nir)

            if valid_matrices:
                completed_matrix = np.vstack(valid_matrices)
                edge_grads.append(np.nanmean(completed_matrix, axis=0))
            else:
                edge_grads.append(np.zeros(config['grad_dim']))

        global_update_1d = server.global_aggregate(edge_grads)

        idx = 0
        for name, param in global_state_dict.items():
            size = param.numel()
            update_tensor = torch.tensor(global_update_1d[idx:idx + size]).reshape(param.shape).to(config['device'])
            global_state_dict[name] -= update_tensor
            idx += size

        mean_acc, mean_loss = evaluate_global_model(clients, global_state_dict, test_loader, config['device'])
        accuracy_history.append(mean_acc)
        loss_history.append(mean_loss)
        pbar.set_postfix(Acc=f"{mean_acc:.2f}%", Loss=f"{mean_loss:.4f}")

    return {"accuracy": accuracy_history, "loss": loss_history}