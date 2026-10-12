import copy
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders

def run_proposed(config, train_loaders=None, test_loader=None, num_classes=None, initial_state=None):
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
    sbs_list = [EdgeSBS(sbs_id, config) for sbs_id in range(config['num_sbs'])]
    lbs = CloudServer(config)

    metric_loss, metric_top1, metric_top5, metric_f1 = [], [], [], []
    pbar = tqdm(range(config['global_rounds']), colour='green', desc="[UAV Multimodal Proposed-FL]")

    for round_idx in pbar:
        global_state_dict = lbs.broadcast_model(global_state_dict)
        edge_updates = []

        for sbs_id, sbs in enumerate(sbs_list):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            delta_matrix, straggler_ids, _ = sbs.collect_gradients(
                cluster_clients, global_state_dict, round_idx
            )
            completed_matrix = sbs.lrmc_complete(delta_matrix, round_idx=round_idx)
            cluster_sum_grad, sbs_samples = sbs.intra_cluster_aggregate(
                completed_matrix, straggler_ids, cluster_clients
            )
            ota_grad = sbs.ota_to_lbs(cluster_sum_grad)
            edge_updates.append((ota_grad, sbs_samples))

        if edge_updates:
            total_samples = sum(s for _, s in edge_updates)
            global_update = sum(g for g, _ in edge_updates) / max(total_samples, 1)
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