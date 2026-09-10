import os
import copy
import json
import torch
from config import get_base_config
from main import run_proposed
from main_fedavg import run_fedavg
from main_fedat import run_fedat
from main_moddrop_fed import run_moddrop_fed
from main_salf import run_salf
from nodes.models import FederatedModel
from utils.dataset import get_dataloaders
from utils.logger import ExperimentLogger
from utils.plotter import plot_results

def execute_experiment(algo_name, config, run_func, train_loaders, test_loader, num_classes, initial_state):
    config['algo_name'] = algo_name
    logger = ExperimentLogger(config, base_dir="result")

    if config.get('use_cache', False):
        cached_data = logger.check_existing_data()
        if cached_data:
            print(f"[{algo_name}] 命中历史运行缓存，直接加载。")
            return cached_data

    print(f"\n=======================================================")
    print(f"   开始执行算法: {algo_name} ({config['dataset']})")
    print(f"=======================================================")

    metrics = run_func(
        config=config,
        train_loaders=train_loaders,
        test_loader=test_loader,
        num_classes=num_classes,
        initial_state=initial_state
    )

    n_rounds = len(metrics.get('loss', []))
    data = {
        "rounds": list(range(1, n_rounds + 1)),
        "loss": metrics["loss"],
        "top1": metrics.get("top1", metrics.get("acc", [])),
        "top5": metrics.get("top5", []),
        "f1": metrics.get("f1", []),
        "acc": metrics.get("acc", metrics.get("top1", []))
    }
    logger.save_data(data)
    return data

if __name__ == "__main__":
    base_config = get_base_config()
    seed = base_config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. 统一加载一次数据划分，确保 5 个算法所见数据完全一致
    print("[Global Environment] 正在生成统一的 Non-IID/IID 数据划分...")
    train_loaders, test_loader, num_classes = get_dataloaders(base_config)

    # 2. 统一初始化模型参数，确保第 0 轮起始权重严格相同
    print("[Global Environment] 正在生成统一的模型初始权重...")
    init_model = FederatedModel(base_config['dataset'], num_classes)
    shared_initial_state = copy.deepcopy(init_model.state_dict())

    # 3. 顺序执行 5 个同维度掉队解决算法
    results_db = {}
    results_db["FedAvg"] = execute_experiment(
        "FedAvg", base_config.copy(), run_fedavg,
        train_loaders, test_loader, num_classes, shared_initial_state
    )

    results_db["FedAT"] = execute_experiment(
        "FedAT", base_config.copy(), run_fedat,
        train_loaders, test_loader, num_classes, shared_initial_state
    )

    results_db["SALF"] = execute_experiment(
        "SALF", base_config.copy(), run_salf,
        train_loaders, test_loader, num_classes, shared_initial_state
    )

    results_db["ModDrop-FedAvg"] = execute_experiment(
        "ModDrop-FedAvg", base_config.copy(), run_moddrop_fed,
        train_loaders, test_loader, num_classes, shared_initial_state
    )

    results_db["Proposed"] = execute_experiment(
        "Proposed", base_config.copy(), run_proposed,
        train_loaders, test_loader, num_classes, shared_initial_state
    )

    # 4. 绘图导出至 ./result/png/ 与 ./result/pdf/
    plot_results(results_db, save_path="result/multimodal_comparison.png")