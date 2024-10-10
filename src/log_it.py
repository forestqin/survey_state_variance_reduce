# -*- coding: utf-8 -*-
# ********************************************************************
# Author: qintao361
# CreateTime: 2024-03-12 17:06:43
# Description: Logger
# Update: Task Update Description
# ********************************************************************

import os, sys
import logging

os.chdir(sys.path[0])

# 配置日志记录器
log_path = './output/state.log'
# log_path = '/home/roi_ds/qintao/01_causalforest_dev/v5/output/causal_forest_subsample.log'
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s [%(levelname)s] %(message)s ',  # 设置日志的格式
    # format='%(asctime)s [%(levelname)s] %(message)s  (%(filename)s:%(lineno)d - %(funcName)s)',  # 设置日志的格式
    handlers=[
        logging.FileHandler(log_path, mode='a'),  # 添加文件处理器，将日志记录到文件中
        logging.StreamHandler()  # 添加流处理器，将日志输出到终端
    ] 
)

# 创建全局 logger 对象
logger = logging.getLogger(__name__)