from config import get_base_config
from utils.logger import ExperimentLogger
from utils.plotter import plot_accuracy_comparison

# 导入各自算法的主入口
from main_fedavg import run_fedavg
from main import run_proposed


def execute_experiment(algo_name, config, run_func):
    """通用的实验执行与缓存包装器"""
    logger = ExperimentLogger(config)
    cached_data = logger.check_existing_data()

    if cached_data:
        print(f"[*] Skipping computation for {algo_name}. Using cached data.")
        return cached_data
    else:
        acc_list = run_func(config)
        data_to_save = {
            "rounds": list(range(config['global_rounds'])),
            "accuracy": acc_list
        }
        logger.save_data(data_to_save)
        return data_to_save


if __name__ == "__main__":

    results_db = {}
    base_cfg = get_base_config()

    # 1. 运行传统 FedAvg
    cfg_fedavg = base_cfg.copy()
    cfg_fedavg['algo_type'] = "Standard_FedAvg"  # 附加标识改变哈希
    results_db["Standard FedAvg"] = execute_experiment(
        "FedAvg", cfg_fedavg, run_fedavg
    )

    # 2. 运行我们的算法
    cfg_proposed = base_cfg.copy()
    cfg_proposed['algo_type'] = "Proposed_LRMC_NOMA"
    results_db["Proposed (NOMA + Sparse + LRMC)"] = execute_experiment(
        "Proposed", cfg_proposed, run_proposed
    )

    # 3. 画图比较
    plot_accuracy_comparison(results_db)