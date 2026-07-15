import numpy as np
from tqdm import tqdm
from nodes.clients import ClientUE
from nodes.edges import EdgeSBS
from nodes.server import CloudServer
from main import get_dataloaders_for_config, get_global_state_dict, create_clients, evaluate_global_model, apply_gradient_update


def run_fedavg(config):
    """运行标准FedAvg算法：丢弃掉队者和信道质量差的客户端，其余梯度直接平均"""
    train_loaders, test_loader, num_classes = get_dataloaders_for_config(config)
    global_state_dict = get_global_state_dict(config, num_classes)
    config['grad_dim'] = sum(p.numel() for p in global_state_dict.values())

    server = CloudServer(config)
    sbs_list = [EdgeSBS(i, config) for i in range(config['num_sbs'])]
    clients = create_clients(config, train_loaders, num_classes)

    accuracy_history, loss_history = [], []
    pbar = tqdm(range(config['global_rounds']), colour='green', desc="[Standard FedAvg]")

    for round_idx in pbar:
        for c in clients:
            c.refresh_state()

        edge_grads = []
        for sbs in sbs_list:
            start = sbs.id * config['clients_per_sbs']
            sbs_clients = clients[start:start + config['clients_per_sbs']]

            # 过滤掉队者和信道质量差的客户端
            valid_grads = [client.local_train(global_state_dict)
                          for client in sbs_clients
                          if not client.is_straggler_state and client.channel_quality >= config['truncation_threshold']]

            if valid_grads:
                edge_grads.append(np.mean(valid_grads, axis=0))

        # 全局聚合并更新模型参数
        apply_gradient_update(global_state_dict, server.global_aggregate(edge_grads), config['device'])

        mean_acc, mean_loss = evaluate_global_model(config, global_state_dict, test_loader, config['device'])
        accuracy_history.append(mean_acc)
        loss_history.append(mean_loss)
        pbar.set_postfix(Acc=f"{mean_acc:.2f}%", Loss=f"{mean_loss:.4f}")

    return {"accuracy": accuracy_history, "loss": loss_history}
