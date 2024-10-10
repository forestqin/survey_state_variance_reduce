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
from src.log_it import logger

os.chdir(sys.path[0])

class Dataset(object):
    def __init__(self, df, pred_y, post_y, random_state=43):
        self.df = df
        self.pred_y = pred_y
        self.post_y = post_y
        self.random_state = random_state

    def get_data_without_effect(self):
        np.random.seed(self.random_state)
        random_T = np.random.randint(0, 2, size=len(self.df))
        X = pd.DataFrame({"const": 1, "t": random_T, "pred_y": self.df[self.pred_y]})
        y = self.df[self.post_y].copy()*1.0
        return X, y

    def get_data_add_effect(self, effect, std):
        X, y = self.get_data_without_effect()
        y.loc[X["t"]==1] += np.random.normal(effect, std)
        return X, y

    def get_data_multiply_effect(self, uplift):
        X, y = self.get_data_without_effect()
        y.loc[X["t"]==1] *= (1 + uplift)
        return X, y


# debug = True
debug = False
if __name__ == "__main__":
    logger.info("===========================Start Experiment=================================")
    df = pd.read_csv("./data/input_data.csv")
    pred_y = "pre_metric"
    post_y = "metric"
    
    N = 1000 if not debug else 1
    effect, std = 700, 10
    uplift = 0.1
    effect_type = "multiply" # "aa", "add", "multiply"
    logger.info(f"基本参数: pred_y: {pred_y}, post_y: {post_y}, N: {N}, effect: {effect}, std: {std}, uplift: {uplift}, effect_type: {effect_type}")

    result = []
    t0 = time.time()
    for random_state in tqdm(range(N)):
        dataset = Dataset(df, pred_y, post_y, random_state)
        if effect_type == "aa":
            X, y = dataset.get_data_without_effect()
        elif effect_type == "add":
            X, y = dataset.get_data_add_effect(effect, std)
        elif effect_type == "multiply":
            X, y = dataset.get_data_multiply_effect(uplift)
        else:
            raise ValueError("effect_type must be 'aa', 'add', or'multiply'")
        
        res = run_one_experiment(X, y, random_state, debug)
        result.extend(res)
    logger.info(f"Experiment Time Spend: {time.time() - t0:.1f}seconds")
    final_df = pd.DataFrame(result)

    # output
    timestamp = datetime.datetime.now().strftime('%m%d%H%M')
    output_path = f"./output/{timestamp}_{post_y}_{N}.csv" if not debug else "./output/debug.csv"
    final_df.to_csv(output_path, index=False)
    logger.info(f"data output path: {output_path}")

    # 计算均值
    cols = ["n", "mean_A", "abs_diff", "rel_diff", "var", "var_reduce", "p_value", "sig"]
    mean_df = final_df.groupby("method")[cols].mean().reset_index() # 计算各列的均值
    logger.info(f"--------mean_df---------\n{mean_df}")

    # 计算方差缩减效果
    var_df = final_df.groupby("method")[cols].var(ddof=1) # 计算各列的方差
    dim_var, cuped_var, state_var = var_df.loc["1_DIM", "abs_diff"], var_df.loc["2_CUPED", "abs_diff"], var_df.loc["3_STATE", "abs_diff"]
    cuped_var_reduce, state_var_reduce = cuped_var / dim_var - 1, state_var / dim_var - 1
    logger.info(f"{dim_var=}, {cuped_var=}, {state_var=}, {cuped_var_reduce=:.2%}, {state_var_reduce=:.2%}\n")


    
    