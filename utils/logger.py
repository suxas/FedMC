import json
import hashlib
import os
import copy
from datetime import datetime

class ExperimentLogger:
    def __init__(self, config, base_dir="result"):
        self.config = config
        self.base_dir = base_dir
        self.algo_name = config.get('algo_name', 'Unknown_Algo')
        self.rounds = config.get('global_rounds', 0)
        self.algo_dir = os.path.join(self.base_dir, "logs", self.algo_name)
        os.makedirs(self.algo_dir, exist_ok=True)

        hash_config = copy.deepcopy(config)
        hash_config.pop('use_cache', None)
        hash_config.pop('device', None)
        self.config_hash = hashlib.md5(json.dumps(hash_config, sort_keys=True).encode('utf-8')).hexdigest()

    def check_existing_data(self):
        if not os.path.exists(self.algo_dir):
            return None
        for filename in os.listdir(self.algo_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.algo_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get("config_hash") == self.config_hash:
                            return data
                except Exception:
                    continue
        return None

    def save_data(self, data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.algo_dir, f"{self.rounds}_{timestamp}.json")
        data["config_hash"] = self.config_hash
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"[Logger] 实验结果已保存: {filepath}")