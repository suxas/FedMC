import numpy as np
from tqdm import tqdm
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from main import get_dataloaders, get_global_state_dict, create_clients, evaluate_global_model, apply_gradient_update


def run_fedavg(config):
    """运行标准FedAvg算法（McMahan et al., 2017）

    原版逻辑：
    1. 每轮随机选择部分客户端
    2. 选中的客户端本地训练
    3. 按数据量加权聚合（不丢弃掉队者，但掉队者不参与训练）
    """
    train_loaders, test_loader, num_classes = get_dataloaders(config)
    global_state_dict = get_global_state_dict(config, num_classes)
    config['grad_dim'] = sum(p.numel() for p in global_state_dict.values())

    server = CloudServer(config)
    clients = create_clients(config, train_loaders, num_classes)
    client_data_sizes = [len(loader.dataset) for loader in train_loaders]

    accuracy_history, loss_history = [], []
    pbar = tqdm(range(config['global_rounds']), colour='green', desc="[FedAvg]")

    for round_idx in pbar:
        for c in clients:
            c.refresh_state(round_idx)

        # 原版FedAvg：每轮随机选择C比例的客户端
        selected_fraction = config.get('client_fraction', 1.0)
        if selected_fraction < 1.0:
            num_selected = max(1, int(len(clients) * selected_fraction))
            selected_clients = np.random.choice(clients, size=num_selected, replace=False).tolist()
        else:
            selected_clients = clients

        # 选中的客户端本地训练（掉队者也参与，但会被标记）
        grads = []
        weights = []
        for client in selected_clients:
            if not client.is_straggler_state:
                grad = client.local_train(global_state_dict)
                grads.append(grad)
                weights.append(client_data_sizes[client.id])

        # 按数据量加权聚合
        if grads:
            total_weight = sum(weights)
            weighted_grad = np.zeros_like(grads[0])
            for g, w in zip(grads, weights):
                weighted_grad += (w / total_weight) * g
            apply_gradient_update(global_state_dict, weighted_grad, config['device'])

        mean_acc, mean_loss = evaluate_global_model(config, global_state_dict, test_loader, config['device'])
        accuracy_history.append(mean_acc)
        loss_history.append(mean_loss)
        pbar.set_postfix(Acc=f"{mean_acc:.2f}%", Loss=f"{mean_loss:.4f}")

    return {"accuracy": accuracy_history, "loss": loss_history}
