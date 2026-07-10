import json
import hashlib
import os
import copy
from datetime import datetime


class ExperimentLogger:
    def __init__(self, config, base_dir="data/results"):
        self.config = config
        self.base_dir = base_dir

        # 获取算法名称和训练轮数
        self.algo_name = self.config.get('algo_name', 'Unknown_Algo')
        self.rounds = self.config.get('global_rounds', 0)

        # 建立各算法专属的文件夹
        self.algo_dir = os.path.join(self.base_dir, self.algo_name)
        os.makedirs(self.algo_dir, exist_ok=True)


        # 拷贝一份配置，防止修改原始的 config
        hash_config = copy.deepcopy(self.config)
        # 将不影响模型结果的控制参数弹出，这样无论 use_cache 是 True 还是 False，Hash 都一样
        hash_config.pop('use_cache', None)
        hash_config.pop('device', None)

        # 计算实验配置的 MD5 Hash
        config_str = json.dumps(hash_config, sort_keys=True)
        self.config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()

    def check_existing_data(self):
        """遍历算法文件夹中的所有文件，比对 config_hash 字段来寻找缓存"""
        if not os.path.exists(self.algo_dir):
            return None

        for filename in os.listdir(self.algo_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.algo_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 检查文件内容中的 hash 是否与当前配置匹配
                        if data.get("config_hash") == self.config_hash:
                            return data
                except Exception as e:
                    continue

        return None

    def save_data(self, data):
        """保存数据，将 Hash 写入内容，文件名使用 轮数+时间"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.rounds}_{timestamp}.json"
        filepath = os.path.join(self.algo_dir, filename)

        # 将 Hash 值写入将要保存的数据字典内容中
        data["config_hash"] = self.config_hash

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        print(f"[Logger] 实验数据已存入: {filepath}")