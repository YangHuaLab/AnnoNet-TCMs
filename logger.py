import sys
import os
import logging
from datetime import datetime

from const import LOG_PATH

# 训练logger
def run_log(
    log_to_file: bool=True,
    prefix: str='',
    suffix: str='',
    log_path: str=LOG_PATH
):
    """log file name example: log_path/{prefix}2023-12-6_10:10:10{suffix}.log

    Args:
        log_to_file (bool, optional): _description_. Defaults to True.
        file_name_prefix (str, optional): _description_. Defaults to ''.
        log_path (str, optional): _description_. Defaults to LOG_PATH.
    """
    logger = logging.getLogger()
    
    for _ in range(len(logger.handlers)):
        logger.removeHandler(logger.handlers[-1])

    # 设置日志格式，创建logger对象，设置StreamHandler日志等级
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    if log_to_file:
        # 创建Filehandler对象，写入.log文件，设置写入文件的日志等级
        file_name = f"{prefix}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{suffix}"
        
        fh = logging.FileHandler(rf"{log_path}{file_name}.log", 'a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # 加载文件到logger对象中
        logger.addHandler(fh)

    # 测试
    # logging.info('info')
    # logging.warning('warning')

def worker_log(level: int=logging.INFO, verbose: bool=True, worker: str=None, msg: str=None):
    """
    levels:
        CRITICAL = 50
        FATAL = CRITICAL
        ERROR = 40
        WARNING = 30
        WARN = WARNING
        INFO = 20
        DEBUG = 10
        NOTSET = 0
    """
    if not verbose:
        return
    
    if worker is None:
        logging.log(level=level, msg=msg)
    else:
        logging.log(level=level, msg=f"worker {worker}: "+msg)
