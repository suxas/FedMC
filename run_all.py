from config import get_base_config
from utils.logger import ExperimentLogger
from utils.plotter import plot_accuracy_comparison
from main_fedavg import run_fedavg
from main import run_proposed


def execute_experiment(algo_name, config, run_func):
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

    print(f"\n=== Hierarchical FL Framework (Device: {base_cfg['device']}) ===")

    cfg_fedavg = base_cfg.copy()
    cfg_fedavg['algo_type'] = "Standard_FedAvg_RealTrain"
    results_db["Standard FedAvg"] = execute_experiment(
        "FedAvg", cfg_fedavg, run_fedavg
    )

    cfg_proposed = base_cfg.copy()
    cfg_proposed['algo_type'] = "Proposed_LRMC_NOMA_RealTrain"
    results_db["Proposed (NOMA + Sparse + LRMC)"] = execute_experiment(
        "Proposed", cfg_proposed, run_proposed
    )

    plot_accuracy_comparison(results_db)