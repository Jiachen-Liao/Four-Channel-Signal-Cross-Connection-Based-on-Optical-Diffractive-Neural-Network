# encoding:utf-8
import logging


def logger(log_filename):
    # 获取根日志记录器
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    # 【优化点】防止重复添加 Handler
    # 如果 log 里面已经有 Handler 了（说明之前初始化过），就不要再加了
    # 否则日志会出现两条、三条...重复的内容
    if not log.handlers:
        # 1. 文件输出
        fh = logging.FileHandler(log_filename)
        # 2. 控制台输出
        sh = logging.StreamHandler()

        # 设置格式
        fm = logging.Formatter('%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s')
        fh.setFormatter(fm)
        sh.setFormatter(fm)

        # 添加到 logger
        log.addHandler(fh)
        log.addHandler(sh)

        log.info('------------The Phase will start.....-------------')

    return log


def output_config(cfg, log):
    # 这个函数非常完美，不用改
    # 它会自动遍历你新 config.json 里的所有键值对（比如 lambda, w0, modes...）
    log.info('---------------- Configuration ----------------')
    for key, val in cfg.items():
        log.info(f'{key}: {val}')
    log.info('-----------------------------------------------')