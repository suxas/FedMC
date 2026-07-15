import json
import hashlib
import os
import copy
from datetime import datetime


class ExperimentLogger:
    """实验日志记录器：支持缓存机制，通过配置哈希避免重复运行"""

    def __init__(self, config, base_dir="data/results"):
        self.config = config
        self.base_dir = base_dir
        self.algo_name = config.get('algo_name', 'Unknown_Algo')
        self.rounds = config.get('global_rounds', 0)

        # 创建算法专属目录
        self.algo_dir = os.path.join(self.base_dir, self.algo_name)
        os.makedirs(self.algo_dir, exist_ok=True)

        # 计算配置哈希（排除不影响结果的参数）
        hash_config = copy.deepcopy(config)
        hash_config.pop('use_cache', None)
        hash_config.pop('device', None)
        self.config_hash = hashlib.md5(json.dumps(hash_config, sort_keys=True).encode('utf-8')).hexdigest()

    def check_existing_data(self):
        """查找哈希匹配的缓存数据"""
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
        """保存实验数据，文件名包含轮数和时间戳"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.algo_dir, f"{self.rounds}_{timestamp}.json")

        data["config_hash"] = self.config_hash
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        print(f"[Logger] 实验数据已存入: {filepath}")