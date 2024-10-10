# -*- coding: utf-8 -*-
# ********************************************************************
# Author: qintao361
# CreateTime: 2024-10-08 17:29
# Description: State算法探索
# ********************************************************************
import os
import sys
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from src.algo import run_one_experiment
from src.simulation_dataset import generate_simulation_dataset
from src.log_it import logger

os.chdir(sys.path[0])

class SimulationDataset(object):
    def __init__(self, df, random_state):
        self.df = df
        self.random_state = random_state
        self.ate = None

    def get_data(self):
        subdf = self.df.sample(frac=0.2, random_state=self.random_state)
        X = pd.DataFrame({"const": 1, "t": subdf["T"], "pred_y": subdf["Pred_Y"]})
        y = subdf["Y"]
        self.ate = subdf["tau_y"].mean()
        return X, y


debug = False
# debug = True
if __name__ == "__main__":
    logger.info("===========================Start Simulation Experiment=================================")
    # 1. 随机生成模拟数据
    N = 5_000_000 if not debug else 10000 # 总样本量
    # tau_y_type = "normal"  # "normal=论文中的tau function"
    tau_y_type = "conditional"  # "conditional=部分用户有effect，部分用户无effect"
    t0 = time.time()
    sim_df = generate_simulation_dataset(N, tau_y_type)
    logger.info(f"Simulation Data {sim_df.shape}, {tau_y_type=}, Generation Time: {time.time() - t0:.1f} seconds")
    
    # 2. 开始模拟实验
    t0 = time.time()
    experiment_num = 1000 if not debug else 1 # 模拟实验的数量
    logger.info(f"基本参数: {experiment_num=}")
    result = []
    ate_list = []
    for random_state in tqdm(range(experiment_num)):
        sim_dataset = SimulationDataset(sim_df, random_state)
        X, y = sim_dataset.get_data()
        res = run_one_experiment(X, y, random_state, debug)
        result.extend(res)
        ate_list.append(sim_dataset.ate)
    logger.info(f"Experiment Time Spend: {time.time() - t0:.1f}seconds")
    final_df = pd.DataFrame(result)

    # 3. 输出结果
    timestamp = datetime.datetime.now().strftime('%m%d%H%M')
    output_path = f"./output/{timestamp}_simulation_{experiment_num}_{tau_y_type}.csv" if not debug else "./output/debug.csv"
    final_df.to_csv(output_path, index=False)
    logger.info(f"data output path: {output_path}")

    # 3.1 计算均值
    cols = ["n", "mean_A", "abs_diff", "rel_diff", "var", "var_reduce", "p_value", "sig"]
    mean_df = final_df.groupby("method")[cols].mean().reset_index() # 计算各列的均值
    logger.info(f"--------mean_df---------\n{mean_df}")
    logger.info(f"True ATE: {np.mean(ate_list)}")

    # 3.2 计算方差缩减效果
    var_df = final_df.groupby("method")[cols].var(ddof=1) # 计算各列的方差
    dim_var, cuped_var, state_var = var_df.loc["1_DIM", "abs_diff"], var_df.loc["2_CUPED", "abs_diff"], var_df.loc["3_STATE", "abs_diff"]
    cuped_var_reduce, state_var_reduce = cuped_var / dim_var - 1, state_var / dim_var - 1
    logger.info(f"{dim_var=}, {cuped_var=}, {state_var=}, {cuped_var_reduce=:.2%}, {state_var_reduce=:.2%}\n")

