import os
import json
from config import get_base_config
from main import run_proposed
from main_fedavg import run_fedavg
from main_salf import run_salf
from utils.logger import ExperimentLogger
from utils.plotter import plot_results


def execute_experiment(algo_name, config, run_func, history_path=None):
    config['algo_name'] = algo_name
    logger = ExperimentLogger(config)

    if history_path:
        if os.path.exists(history_path):
            print(f"[{algo_name}] 加载历史数据: {history_path}")
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)

    if config.get('use_cache', False):
        cached_data = logger.check_existing_data()
        if cached_data:
            print(f"[{algo_name}] 使用缓存数据")
            return cached_data

    print(f"\n--- 开始执行: {algo_name} ({config['dataset']}) ---")
    metrics = run_func(config)
    eval_every = config.get('eval_every', 1)
    n_evals = len(metrics.get('loss', []))
    data = {
        "rounds": [r * eval_every for r in range(n_evals)],
        "loss": metrics.get("loss", [])
    }
    # SYSU-MM01 输出 R1/CMC/mAP，MNIST/CIFAR10 输出 accuracy
    if "R1" in metrics:
        data["R1"] = metrics["R1"]
        data["CMC"] = metrics["CMC"]
        data["mAP"] = metrics["mAP"]
    else:
        data["accuracy"] = metrics.get("accuracy", [])

    logger.save_data(data)
    return data


if __name__ == "__main__":
    base_config = get_base_config()
    results_db = {}

    results_db["FedAvg"] = execute_experiment("FedAvg", base_config.copy(), run_fedavg)
    results_db["SALF"] = execute_experiment("SALF", base_config.copy(), run_salf)
    results_db["Proposed"] = execute_experiment("Proposed", base_config.copy(), run_proposed)

    plot_results(results_db)
