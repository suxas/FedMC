import json
import hashlib
import os

class ExperimentLogger:
    def __init__(self, config, save_dir="data/results"):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.config_hash = self._generate_hash()
        self.filepath = os.path.join(self.save_dir, f"exp_{self.config_hash}.json")

    def _generate_hash(self):
        # 确保字典排序一致，生成唯一 MD5 哈希
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()

    def check_existing_data(self):
        if os.path.exists(self.filepath):
            print(f"[Logger] Found cached simulation data for hash: {self.config_hash}")
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return None

    def save_data(self, data):
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[Logger] Simulation data securely saved to {self.filepath}")