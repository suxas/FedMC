import os
import json
from config import get_base_config
from main import run_proposed
from main_fedavg import run_fedavg
from main_salf import run_salf
from utils.logger import ExperimentLogger
from utils.plotter import plot_results


def execute_experiment(algo_name, config, run_func, history_path=None):
    """执行实验：支持缓存和强制加载历史数据"""
    config['algo_name'] = algo_name
    logger = ExperimentLogger(config)

    # 优先加载指定历史数据
    if history_path:
        if os.path.exists(history_path):
            print(f"[{algo_name}] 强制加载指定历史数据: {history_path}")
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        print(f"[{algo_name}] 警告: 指定的历史文件未找到 ({history_path})，将降级处理...")

    # 检查缓存
    if config.get('use_cache', False):
        cached_data = logger.check_existing_data()
        if cached_data:
            print(f"[{algo_name}] 发现哈希匹配的缓存数据，跳过计算。")
            return cached_data

    # 执行算法并保存结果
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

    # 运行三个算法
    results_db["Standard FedAvg"] = execute_experiment("FedAvg", base_config.copy(), run_fedavg)
    results_db["SALF (Layer-Wise)"] = execute_experiment("SALF", base_config.copy(), run_salf)
    results_db["Proposed (Multi-modal LRMC)"] = execute_experiment("Proposed_LRMC", base_config.copy(), run_proposed)

    plot_results(results_db)