# -*- coding:utf-8 -*-
import os, os.path as osp
import time

def strftime(t = None):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t or time.time()))   # 时间格式

#################
# Logging
#################
import logging
from logging.handlers import TimedRotatingFileHandler
# 为日志模块配置基本信息。设置后可以直接使用logging来打印日志
logging.basicConfig(format="[ %(asctime)s][%(module)s.%(funcName)s] %(message)s")   # 格式：[时间][模块.函数] 信息

DEFAULT_LEVEL = logging.INFO    # 设置日志等级
DEFAULT_LOGGING_DIR = osp.join("logs", "gcforest")    # 默认日志路径
fh = None


# 初始化函数，确保fh被赋值
def init_fh():
    global fh
    if fh is not None:
        return
    if DEFAULT_LOGGING_DIR is None:
        return
    if not osp.exists(DEFAULT_LOGGING_DIR):   # 确保路径存在
        os.makedirs(DEFAULT_LOGGING_DIR)
    logging_path = osp.join(DEFAULT_LOGGING_DIR, strftime() + ".log")   # 日志名，以时间命名
    fh = logging.FileHandler(logging_path)   # 创造一个filehandler
    fh.setFormatter(logging.Formatter("[ %(asctime)s][%(module)s.%(funcName)s] %(message)s"))

# 更新日志等级
def update_default_level(defalut_level):
    global DEFAULT_LEVEL
    DEFAULT_LEVEL = defalut_level

# 更新日志目录
def update_default_logging_dir(default_logging_dir):
    global DEFAULT_LOGGING_DIR
    DEFAULT_LOGGING_DIR = default_logging_dir

# 获得日志  有level就设置一个新的等级
def get_logger(name="gcforest", level=None):
    level = level or DEFAULT_LEVEL
    logger = logging.getLogger(name)
    logger.setLevel(level)
    init_fh()
    if fh is not None:
        logger.addHandler(fh)
    return logger
