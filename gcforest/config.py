from .data_cache import DataCache


# 训练配置
class GCTrainConfig(object):
    def __init__(self, train_config):
        self.keep_model_in_mem = train_config.get("keep_model_in_mem", True)     # 是否将模型保存在内存
        self.random_state = train_config.get("random_state", 0)   # 随机数的种子
        self.model_cache_dir = strip(train_config.get("model_cache_dir", None))    # 存储在内存中的位置
        self.data_cache = DataCache(train_config.get("data_cache", {}))
        self.phases = train_config.get("phases", ["train", "test"])

        for data_name in ("X", "y"):
            if data_name not in self.data_cache.config["keep_in_mem"]:    # data_name是否在字典中，不在就将其设为true（存入内存）
                self.data_cache.config["keep_in_mem"][data_name] = True
            if data_name not in self.data_cache.config["cache_in_disk"]:  # data_name是否在字典中，不在就将其设为false（不存入磁盘）
                self.data_cache.config["cache_in_disk"][data_name] = False


def strip(s):    # 去除首尾空格
    if s is None:
        return None
    s = s.strip()
    if len(s) == 0:
        return None
    return s
