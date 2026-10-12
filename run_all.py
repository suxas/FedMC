import os
import copy
import random
import numpy as np
import torch
from config import get_base_config
from main import run_proposed
from main_fedat import run_fedat
from main_moddrop_fed import run_moddrop_fed
from main_salf import run_salf
from main_fedgc import run_fedgc
from nodes.models import FederatedModel
from utils.dataset import get_dataloaders
from utils.logger import ExperimentLogger
from utils.plotter import plot_results

def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def execute_experiment(algo_name, config, run_func, initial_state):
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

    seed = config.get("seed", 42)
    set_deterministic_seed(seed)
    train_loaders, test_loader, num_classes = get_dataloaders(config)

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
    master_seed = base_config.get("seed", 42)
    set_deterministic_seed(master_seed)

    print("[Global Environment] 正在生成统一的模型初始权重...")
    _, _, num_classes = get_dataloaders(base_config)
    init_model = FederatedModel(base_config['dataset'], num_classes)
    shared_initial_state = copy.deepcopy(init_model.state_dict())


    results_db = {}

    results_db["FedAT"] = execute_experiment(
        "FedAT", base_config.copy(), run_fedat, shared_initial_state
    )

    results_db["SALF"] = execute_experiment(
        "SALF", base_config.copy(), run_salf, shared_initial_state
    )

    results_db["ModDrop-FedAvg"] = execute_experiment(
        "ModDrop-FedAvg", base_config.copy(), run_moddrop_fed, shared_initial_state
    )

    results_db["FedGC"] = execute_experiment(
        "FedGC", base_config.copy(), run_fedgc, shared_initial_state
    )

    results_db["Proposed"] = execute_experiment(
        "Proposed", base_config.copy(), run_proposed, shared_initial_state
    )

    plot_results(results_db, save_path="result/multimodal_comparison.png")