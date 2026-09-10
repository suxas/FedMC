import copy
import numpy as np
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.server import CloudServer
from utils.dataset import get_dataloaders

def get_modality_split_indices(global_state_dict):
    vis_count = 0
    tot_count = 0
    for name, tensor in global_state_dict.items():
        if not tensor.is_floating_point():
            continue
        n = tensor.numel()
        if 'visual' in name:
            vis_count += n
        tot_count += n
    return vis_count, tot_count

def run_moddrop_fed(config, train_loaders=None, test_loader=None, num_classes=None, initial_state=None):
    if train_loaders is None or test_loader is None:
        train_loaders, test_loader, num_classes = get_dataloaders(config)

    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    if initial_state is not None:
        global_model.load_state_dict(copy.deepcopy(initial_state))
    global_state_dict = global_model.state_dict()

    clients = [
        ClientUE(i, config, train_loaders[i], None, num_classes)
        for i in range(config['num_sbs'] * config['clients_per_sbs'])
    ]
    lbs = CloudServer(config)
    straggler_ratio = config.get('straggler_ratio', 0.5)
    vis_split, total_params = get_modality_split_indices(global_state_dict)

    metric_loss, metric_top1, metric_top5, metric_f1 = [], [], [], []
    pbar = tqdm(range(config['global_rounds']), colour='yellow', desc="[ModDrop-FedAvg]")

    for round_idx in pbar:
        edge_updates = []
        for sbs_id in range(config['num_sbs']):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            times = [c.simulate_communication_time(round_idx, drop_visual=False)[1] for c in cluster_clients]
            k = max(1, int(len(cluster_clients) * (1.0 - straggler_ratio))) - 1
            t_win = sorted(times)[k]

            sbs_grad_sum = np.zeros(total_params, dtype=np.float32)
            sbs_mask_weight = np.zeros(total_params, dtype=np.float32)

            for client, t in zip(cluster_clients, times):
                n_samples = client.num_train_samples
                if t <= t_win:
                    # 准时节点: 运行全双模态网络
                    grad, _ = client.local_train(global_state_dict, drop_visual=False)
                    mask = np.ones(total_params, dtype=np.float32)
                else:
                    # 掉队节点: 触发 Modality Dropout，特征填零只训练文本+分类头
                    grad, _ = client.local_train(global_state_dict, drop_visual=True)
                    mask = np.zeros(total_params, dtype=np.float32)
                    mask[vis_split:] = 1.0  # 视觉部分权重为 0，不上传不更新

                sbs_grad_sum += grad * n_samples
                sbs_mask_weight += mask * n_samples

            if np.sum(sbs_mask_weight) > 0:
                edge_updates.append((sbs_grad_sum, sbs_mask_weight))

        if edge_updates:
            global_grad = sum(g for g, _ in edge_updates)
            global_weight = sum(w for _, w in edge_updates)
            global_update = global_grad / np.maximum(global_weight, 1.0)
            lbs.apply_update(global_state_dict, global_update, config['device'])

        eval_res = lbs.evaluate(global_state_dict, test_loader, global_model, config['device'])
        metric_loss.append(eval_res['loss'])
        metric_top1.append(eval_res['top1'])
        metric_top5.append(eval_res['top5'])
        metric_f1.append(eval_res['f1'])

        pbar.set_postfix(
            Loss=f"{eval_res['loss']:.4f}",
            Top1=f"{eval_res['top1']:.2f}%",
            Top5=f"{eval_res['top5']:.2f}%",
            F1=f"{eval_res['f1']:.2f}%"
        )

    return {
        "loss": metric_loss,
        "top1": metric_top1,
        "top5": metric_top5,
        "f1": metric_f1,
        "acc": metric_top1
    }