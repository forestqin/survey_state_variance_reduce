import numpy as np
import pandas as pd
import random
import xgboost as xgb
import math
import time
from scipy.special import gamma, digamma
import matplotlib.pyplot as plt


class VarianceReduction:

    def __init__(self, feature_dim=5, data_size=200000, outlier_frac=0.002, files=None):

        if files is None:
            self.df = data_generation(
                feature_dim, data_size, outlier_frac)
        else:
            self.df = read_data(files)
        self.ate = self.df["true_ite"].mean()


def data_generation(feature_dim, data_size, outlier_frac):
    clean_num = int(data_size*(1 - outlier_frac))
    outlier_num = int(data_size * outlier_frac)

    np.random.seed(0)
    var_mean = np.random.choice(range(10), size=feature_dim)
    var_covariance = np.eye(feature_dim)

    np.random.seed(1)
    var_data = np.random.multivariate_normal(
        var_mean, var_covariance, clean_num)

    # add noise
    np.random.seed(2)
    noise0 = np.random.normal(loc=0.0, scale=25.0, size=clean_num)
    noise1 = np.random.normal(loc=0.0, scale=25.0, size=clean_num)
    # noise0, noise1

    real_ite = var_data[:, 0] * var_data[:, 2] + \
        np.log(1 + np.exp(var_data[:, 1]))
    ATE = real_ite.mean()

    b_X = 10 * np.sin(math.pi * var_data[:, 0] * var_data[:, 1]) + 6 * np.power(
        var_data[:, 2], 2) + 10 * np.abs(var_data[:, 3]) + 5 * np.abs(var_data[:, 4]) + 50
    y0 = b_X + noise0
    y1 = b_X + real_ite + noise1

    clean_df = pd.DataFrame(var_data, columns=['f0', 'f1', 'f2', 'f3', 'f4'])
    clean_df["y0"] = y0
    clean_df["y1"] = y1
    clean_df["constant"] = 1
    clean_df["true_ite"] = real_ite
    clean_df["flag"] = 1

    np.random.seed(3)
    outlier_x = np.random.multivariate_normal(
        var_mean, var_covariance, outlier_num)
    outlier_y0 = np.random.choice(range(int(np.mean(
        y1) + 4*np.std(y1)), int(np.mean(y1) + 20*np.std(y1))), size=outlier_num)
    outlier_y1 = np.random.choice(range(int(np.mean(
        y1) + 4*np.std(y1)), int(np.mean(y1) + 20*np.std(y1))), size=outlier_num)
    outlier_df = pd.DataFrame(
        outlier_x, columns=['f0', 'f1', 'f2', 'f3', 'f4'])

    outlier_df["y0"] = outlier_y0
    outlier_df["y1"] = outlier_y1
    outlier_df["constant"] = 1
    outlier_df["true_ite"] = 0
    outlier_df["flag"] = 0
    full_df = pd.concat([clean_df, outlier_df]).sample(
        frac=1.0, random_state=4).reset_index(drop=True)
    return full_df


def read_data(files):
    print(time.strftime("%Y-%m-%d %H:%M:%S",
          time.localtime()), ": start read data...")

    dfs = []
    for file in files:
        df = pd.read_csv(file, sep='\t')
        dfs.append(df)

    df_randomized = pd.concat(dfs, ignore_index=True)
    df_randomized = df_randomized.reset_index(drop=True)
    return df_randomized


def ml_train(df):
    d1 = df.sample(frac=0.5)
    d2 = df[~df.index.isin(d1.index)]
    train_feature = ["f0", "f1", "f2", "f3", "f4"]
    xgb_1 = xgb.XGBRegressor(n_jobs=16, objective='reg:squarederror', colsample_bytree=0.9, learning_rate=0.1,
                             max_depth=3, alpha=10, n_estimators=80)
    X_train1 = d1[train_feature].values
    metric_label1 = ((d1["y0"]+d1["y1"])/2).values.flatten()
    xgb_1.fit(X_train1, metric_label1)

    xgb_2 = xgb.XGBRegressor(n_jobs=16, objective='reg:squarederror', colsample_bytree=0.9, learning_rate=0.1,
                             max_depth=3, alpha=10, n_estimators=80)
    X_train2 = d2[train_feature].values
    metric_label2 = ((d2["y0"]+d2["y1"])/2).values.flatten()
    xgb_2.fit(X_train2, metric_label2)

    Y_pred_2 = xgb_1.predict(X_train2)
    Y_pred_1 = xgb_2.predict(X_train1)
    d1["metric_pred"] = Y_pred_1
    d2["metric_pred"] = Y_pred_2
    df_ml = pd.concat([d1, d2])
    return df_ml


def STATE(array_ab, gv=10):
    # EM parameter initialization
    a0, a1, a2, sigmma = 10, 10, 10, 100
    treatment, proxy, y = array_ab[:, 1], array_ab[:, 2], array_ab[:, 3]
    sample_size = array_ab.shape[0]

    for iter in range(10000):

        params = [a0, a1, a2]
        # E-step: calculate the posterior distribution of latent variables
        ga = gv / 2 + 1 / 2
        gb = gv / 2 + (y - a0 - a1*treatment - a2*proxy)**2 / (2 * sigmma)
        gu = ga / gb

        # M-step: derive parameters
        sum0 = sum((y - a1*treatment - a2*proxy) * gu)
        sum00 = sum(gu)
        a0 = sum0 / sum00

        sum1 = sum((y - a0 - a2*proxy)*treatment * gu)
        sum11 = sum(gu * treatment**2)
        a1 = sum1 / sum11

        sum2 = sum((y - a0 - a1*treatment)*proxy * gu)
        sum22 = sum(gu * proxy**2)
        a2 = sum2 / sum22

        sum7 = sum((y - a0 - a1*treatment - a2*proxy)**2 * gu)
        sigmma = sum7 / sample_size

        if np.allclose(params, [a0, a1, a2], atol=1e-4):
            # print('{}th iteration，a0:{:.5f}, a1:{:.5f}, a2:{:.5f}, sigmma:{:.5f}'.format(iter, a0, a1, a2, sigmma))
            print('{}th iteration STATE:{:.5f}'.format(iter, a1))
            break
    return a1


def eval_ate(df_population, simulations_num, sample_size):
    dim_abate_list, cuped_abate_list, cupac_abate_list, mlrate_abate_list, state_abate_list = [], [], [], [], []
    cuped_feature, cupac_feature = "f2", "metric_pred"
    cuped_mean = df_population[cuped_feature].mean()
    cupac_mean = df_population[cupac_feature].mean()

    cov_cuped = np.cov(df_population[cuped_feature], df_population["y0"])
    theta_cuped = cov_cuped[0, 1] / cov_cuped[0, 0]
    cov_cupac = np.cov(df_population[cupac_feature], df_population["y0"])
    theta_cupac = cov_cupac[0, 1] / cov_cupac[0, 0]

    n = df_population.shape[0]
    for i in range(1, simulations_num+1):
        print("-"*50)
        print('{}th simulation'.format(i))

        df_ab = df_population.sample(sample_size, random_state=i)

        np.random.seed(i)
        random_array = np.random.choice(range(2), sample_size)

        df_ab["treat"] = random_array
        df_ab["metric_view"] = df_ab["treat"] * \
            df_ab["y1"] + (1-df_ab["treat"])*df_ab["y0"]
        df_ab["metric_cuped"] = df_ab["metric_view"] - \
            (df_ab["f2"] - cuped_mean) * theta_cuped
        df_ab["metric_cupac"] = df_ab["metric_view"] - \
            (df_ab["metric_pred"] - cupac_mean) * theta_cupac

        df_ab_agg = df_ab.groupby(["treat"]).agg(
            {"metric_view": "mean", "metric_cuped": "mean", "metric_cupac": "mean"})

        # DIM_ATE
        dim_ate = df_ab_agg.loc[1, "metric_view"] - \
            df_ab_agg.loc[0, "metric_view"]
        dim_abate_list.append(dim_ate)
        print("DIM_ATE:{:.5f}".format(dim_ate))

        # CUPED_ATE
        cuped_ate = df_ab_agg.loc[1, "metric_cuped"] - \
            df_ab_agg.loc[0, "metric_cuped"]
        cuped_abate_list.append(cuped_ate)
        print("CUPED_ATE:{:.5f}".format(cuped_ate))

        # CUPAC_ATE
        cupac_ate = df_ab_agg.loc[1, "metric_cupac"] - \
            df_ab_agg.loc[0, "metric_cupac"]
        cupac_abate_list.append(cupac_ate)
        print("CUPAC_ATE:{:.5f}".format(cupac_ate))

        # MLRATE
        df_ab["cross"] = (df_ab["metric_pred"] - cupac_mean) * df_ab["treat"]
        array_ab = df_ab[["constant", "treat",
                          "metric_pred", "cross", "metric_view"]].values
        matrix_ab = array_ab[:, :4]
        y = array_ab[:, 4]

        ols_para_iter = np.dot(np.linalg.inv(
            np.dot(matrix_ab.T, matrix_ab)), np.dot(matrix_ab.T, y))
        mlrate = ols_para_iter[1]
        mlrate_abate_list.append(mlrate)
        print("MLRATE:{:.5f}".format(mlrate))

        # STATE
        state = STATE(array_ab[:, [0, 1, 2, 4]], 10)
        state_abate_list.append(state)

    return dim_abate_list, cuped_abate_list, cupac_abate_list, mlrate_abate_list, state_abate_list


def Emp_Cov(ate_list, target):
    cover_cnt, simulations_num = 0, len(ate_list)
    for i in range(simulations_num):
        if np.abs(ate_list[i]-target) <= 1.96*np.std(ate_list):
            cover_cnt += 1
    return cover_cnt / simulations_num


def Variance(ate_list):
    return np.var(ate_list)


if __name__ == '__main__':
    evaluate = VarianceReduction()
    df_ml = ml_train(evaluate.df)
    dim_abate_list, cuped_abate_list, cupac_abate_list, mlrate_abate_list, state_abate_list = eval_ate(df_ml,2,20000)
    ## variance 
    DIM_VAR = Variance(dim_abate_list)
    CUPED_VAR = Variance(cuped_abate_list)
    CUPAC_VAR = Variance(cupac_abate_list)
    MLRATE_VAR = Variance(mlrate_abate_list)
    STATE_VAR = Variance(state_abate_list)
    print(f"{DIM_VAR=:.5f}, {CUPED_VAR=:.5f}, {CUPAC_VAR=:.5f}, {MLRATE_VAR=:.5f}, {STATE_VAR=:.5f}")

    # Empirical coverage
    DIM_EMP_COV = Emp_Cov(dim_abate_list, evaluate.ate)
    CUPED_EMP_COV = Emp_Cov(cuped_abate_list, evaluate.ate)
    CUPAC_EMP_COV = Emp_Cov(cupac_abate_list, evaluate.ate)
    MLRATE_EMP_COV = Emp_Cov(mlrate_abate_list, evaluate.ate)
    STATE_EMP_COV = Emp_Cov(state_abate_list, evaluate.ate)
    print(f"{DIM_EMP_COV=:.2%}, {CUPED_EMP_COV=:.2%}, {CUPAC_EMP_COV=:.2%}, {MLRATE_EMP_COV=:.2%}, {STATE_EMP_COV=:.2%}")