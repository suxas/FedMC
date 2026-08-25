import numpy as np
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.server import CloudServer
from utils.dataset import get_dataloaders


def _salf_partial(model, full_grad):
    sizes = [p.numel() for p in model.parameters()]
    n_layers = len(sizes)
    keep = np.random.randint(1, max(2, n_layers // 2))
    cut = sum(sizes[:n_layers - keep])
    g = full_grad.copy()
    if cut > 0:
        g[:cut] = 0.0
    return g


def run_salf(config):
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = [
        ClientUE(i, config, train_loaders[i], None, num_classes)
        for i in range(config['num_sbs'] * config['clients_per_sbs'])
    ]
    lbs = CloudServer(config)
    straggler_ratio = config.get('straggler_ratio', 0.5)

    metric_acc, metric_loss = [], []
    pbar = tqdm(range(config['global_rounds']), colour='blue', desc="[SALF]")

    for round_idx in pbar:
        edge_updates = []

        for sbs_id in range(config['num_sbs']):
            start_idx = sbs_id * config['clients_per_sbs']
            end_idx = start_idx + config['clients_per_sbs']
            cluster_clients = clients[start_idx:end_idx]

            times = [c.simulate_communication_time(round_idx)[1] for c in cluster_clients]
            k = max(1, int(len(cluster_clients) * (1.0 - straggler_ratio))) - 1
            t_win = sorted(times)[k]

            sbs_sum_grad = 0.0
            sbs_samples = 0
            for client, t in zip(cluster_clients, times):
                grad, _ = client.local_train(global_state_dict)
                if t > t_win:
                    grad = _salf_partial(global_model, grad)
                sbs_sum_grad += grad * client.num_train_samples
                sbs_samples += client.num_train_samples

            if sbs_samples > 0:
                edge_updates.append((sbs_sum_grad, sbs_samples))

        if edge_updates:
            total_samples = sum(s for _, s in edge_updates)
            global_update = sum(g for g, _ in edge_updates) / max(total_samples, 1)
            lbs.apply_update(global_state_dict, global_update, config['device'])

        acc, loss = lbs.evaluate(global_state_dict, test_loader, global_model, config['device'])
        metric_acc.append(acc)
        metric_loss.append(loss)
        pbar.set_postfix(Loss=f"{loss:.4f}", Acc=f"{acc:.2f}%")

    return {"loss": metric_loss, "acc": metric_acc}