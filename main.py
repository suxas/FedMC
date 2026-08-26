from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from utils.dataset import get_dataloaders


def run_proposed(config):
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = [
        ClientUE(i, config, train_loaders[i], None, num_classes)
        for i in range(config['num_sbs'] * config['clients_per_sbs'])
    ]
    sbs_list = [EdgeSBS(sbs_id, config) for sbs_id in range(config['num_sbs'])]
    lbs = CloudServer(config)

    metric_acc, metric_loss = [], []
    pbar = tqdm(range(config['global_rounds']), colour='green', desc="[Multimodal LRMC-FL]")

    for round_idx in pbar:
        global_state_dict = lbs.broadcast_model(global_state_dict)
        edge_updates = []

        for sbs_id, sbs in enumerate(sbs_list):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            grad_matrix, straggler_ids, _ = sbs.collect_gradients(
                cluster_clients, global_state_dict, round_idx
            )
            completed_matrix = sbs.lrmc_complete(grad_matrix)
            cluster_sum_grad, sbs_samples = sbs.intra_cluster_aggregate(
                completed_matrix, straggler_ids, cluster_clients
            )
            ota_grad = sbs.ota_to_lbs(cluster_sum_grad)
            edge_updates.append((ota_grad, sbs_samples))

        if edge_updates:
            total_samples = sum(s for _, s in edge_updates)
            global_update = sum(g for g, _ in edge_updates) / max(total_samples, 1)
            lbs.apply_update(global_state_dict, global_update, config['device'])

        acc, loss = lbs.evaluate(global_state_dict, test_loader, global_model, config['device'])
        metric_acc.append(acc)
        metric_loss.append(loss)
        pbar.set_postfix(Loss=f"{loss:.4f}", Acc=f"{acc:.2f}%")

    return {"loss": metric_loss, "acc": metric_acc}