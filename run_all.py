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

    if history_path and os.path.exists(history_path):
        print(f"[{algo_name}] 强制加载指定历史数据: {history_path}")
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif history_path:
        print(f"[{algo_name}] 警告: 指定的历史文件未找到 ({history_path})，将降级处理...")

    if config.get('use_cache', False):
        cached_data = logger.check_existing_data()
        if cached_data:
            print(f"[{algo_name}] 发现哈希匹配的缓存数据，跳过计算。")
            return cached_data

    print(f"\n--- 开始执行算法: {algo_name} ---")
    metrics = run_func(config)

    data = {
        "rounds": list(range(config['global_rounds'])),
        "accuracy": metrics["accuracy"],
        "loss": metrics["loss"]
    }
    logger.save_data(data)
    return data


if __name__ == "__main__":
    base_config = get_base_config()
    results_db = {}

    cfg_fedavg = base_config.copy()
    results_db["Standard FedAvg"] = execute_experiment(
        "FedAvg", cfg_fedavg, run_fedavg,
        #history_path="data/results/FedAvg/30_20260708_150805.json"
    )

    cfg_salf = base_config.copy()
    results_db["SALF (Layer-Wise)"] = execute_experiment(
        "SALF", cfg_salf, run_salf,
    )

    cfg_proposed = base_config.copy()
    results_db["Proposed (Multi-modal LRMC)"] = execute_experiment(
        "Proposed_LRMC", cfg_proposed, run_proposed,
        #history_path="data/results/Proposed_LRMC/30_20260708_153213.json"
    )

    plot_results(results_db)