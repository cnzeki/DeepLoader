#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from logging import handlers
import os
import time


MAX_BYTES   = 5e7 # 最大50MB
MAX_COUNTS  = 3   # 最多三个文件


class Loggers:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
            # cls.__instance.__init__(*args, **kwargs)
        return cls.__instance

    def __init__(self, tag='test', logdir='.'):
        # 设置输出格式
        # fmt = [%(asctime)s]-[%(levelname)s]-[%(filename)s]-[%(funcName)s:%(lineno)d] : %(message)s'
        formater = logging.Formatter(
            '[%(asctime)s]-[%(levelname)s]-[%(filename)s:%(lineno)d] : %(message)s')
        # 定义一个日志收集器
        self.logger = logging.getLogger('log')
        # 设定级别
        self.logger.setLevel(logging.DEBUG)
        # 输出渠道一 - 文件形式
        #uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime())
        try:
            os.makedirs(logdir)
        except:
            pass
        filename = logdir + '/' + tag + '.log'
        print('Logging to file:%s' % filename)
        self.fileLogger = handlers.RotatingFileHandler(filename, maxBytes=MAX_BYTES, backupCount=MAX_COUNTS)

        # 输出渠道二 - 控制台
        self.console = logging.StreamHandler()
        # 控制台输出级别
        self.console.setLevel(logging.DEBUG)
        # 输出渠道对接输出格式
        self.console.setFormatter(formater)
        self.fileLogger.setFormatter(formater)
        # 日志收集器对接输出渠道
        self.logger.addHandler(self.fileLogger)
        self.logger.addHandler(self.console)

    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def excepiton(self, *args, **kwargs):
        self.logger.exception(*args, **kwargs)


if __name__ == '__main__':
    loggers = Loggers(tag='test', logdir='log')
    loggers.debug('debug')
    loggers.info('info')
    loggers.warn('warn')
