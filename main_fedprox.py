from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.models import FederatedModel
from nodes.server import CloudServer
from utils.dataset import get_dataloaders


def run_fedprox(config):
    """
    多模态 FedProx 算法:
      - 正常节点: 完成全部 local_epochs
      - 掉队节点: 执行部分训练 (Partial Work: local_epochs // 2)
      - 所有节点添加 Proximal 正则化约束防止掉队漂移
    """
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    global_model = FederatedModel(config['dataset'], num_classes).to(config['device'])
    global_state_dict = global_model.state_dict()

    clients = [
        ClientUE(i, config, train_loaders[i], None, num_classes)
        for i in range(config['num_sbs'] * config['clients_per_sbs'])
    ]
    lbs = CloudServer(config)
    straggler_ratio = config.get('straggler_ratio', 0.5)
    mu = config.get('fedprox_mu', 0.01)
    full_epochs = config.get('local_epochs', 3)
    partial_epochs = max(1, full_epochs // 2)

    metric_acc, metric_loss = [], []
    pbar = tqdm(range(config['global_rounds']), colour='magenta', desc="[Multimodal FedProx]")

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
                if t <= t_win:
                    # 正常客户端：完整轮次训练 + 近端约束
                    grad, _ = client.local_train(global_state_dict, mu=mu, epochs=full_epochs)
                else:
                    # 掉队客户端：执行 Partial Work 训练 + 近端约束
                    grad, _ = client.local_train(global_state_dict, mu=mu, epochs=partial_epochs)

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