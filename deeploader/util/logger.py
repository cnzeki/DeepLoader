#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from logging import handlers
import os
import time


MAX_BYTES   = 5e7 # 最大50MB
MAX_COUNTS  = 3   # 最多三个文件

_logger_map = dict()


def getLogger(tag='test', logdir='.'):
    if tag in _logger_map:
        print('Return cached logger: %s' % tag)
        return _logger_map[tag]
        
    # 设置输出格式
    # fmt = [%(asctime)s]-[%(levelname)s]-[%(filename)s]-[%(funcName)s:%(lineno)d] : %(message)s'
    formater = logging.Formatter(
        '[%(asctime)s]-[%(levelname)s]-[%(filename)s:%(lineno)d] : %(message)s')
    # 定义一个日志收集器
    logger = logging.getLogger(tag)
    # 设定级别
    logger.setLevel(logging.DEBUG)
    # 输出渠道一 - 文件形式
    # uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime())
    try:
        os.makedirs(logdir)
    except:
        pass
    filename = logdir + '/' + tag + '.log'
    print('Logging to file:%s' % filename)
    fileLogger = handlers.RotatingFileHandler(filename, maxBytes=MAX_BYTES, backupCount=MAX_COUNTS)

    # 输出渠道二 - 控制台
    console = logging.StreamHandler()
    # 控制台输出级别
    console.setLevel(logging.DEBUG)
    # 输出渠道对接输出格式
    console.setFormatter(formater)
    fileLogger.setFormatter(formater)
    # 日志收集器对接输出渠道
    logger.addHandler(fileLogger)
    logger.addHandler(console)
    logger.debug('Logger %s started.' % tag)
    _logger_map[tag] = logger
    return logger


if __name__ == '__main__':
    logger = getLogger(tag='test', logdir='log')
    logger.debug('debug')
    logger.info('info')
    logger.warn('warn')
