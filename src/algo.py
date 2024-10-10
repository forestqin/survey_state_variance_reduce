
# ********************************************************************
# Author: qintao361
# CreateTime: 2024-10-08 17:29
# Description: State算法探索
# ********************************************************************

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import psi
from scipy.optimize import fsolve
from scipy.special import gamma
from functools import partial
from scipy import stats

class Dim(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def run(self):
        A = self.y[self.X["t"]==0]
        B = self.y[self.X["t"]==1]
        mean_A, mean_B = np.mean(A), np.mean(B)
        a1 = mean_B - mean_A
        var = np.var(A, ddof=1)/len(A) + np.var(B, ddof=1)/len(B)
        t_value, p_value = stats.ttest_ind(B, A, equal_var=True)
        sig = 1 if p_value <= 0.05 else 0
        return a1, mean_A, mean_B, var, p_value, sig


class Cuped(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def run(self):
        matrix = np.cov(self.X["pred_y"], self.y)
        pre_var, cov = matrix[0][0], matrix[0][1]
        theta = cov * 1.0 / pre_var
        cuped_y = self.y - theta * self.X["pred_y"]
        cuped_A = cuped_y[self.X["t"]==0]
        cuped_B = cuped_y[self.X["t"]==1]
        a1 = cuped_B.mean() - cuped_A.mean()

        t_value, p_value = stats.ttest_ind(cuped_B, cuped_A, equal_var=True)
        sig = 1 if p_value <= 0.05 else 0

        cuped_var = np.var(cuped_A, ddof=1)/len(cuped_A) + np.var(cuped_B, ddof=1)/len(cuped_B)
        return a1, cuped_var, p_value, sig

    def run2(self):
        '''
        该方法和run在a1估计上是等效的
        '''
        model = sm.OLS(self.y, self.X).fit()  # 拟合线性回归模型
        a0, a1, a2 = model.params["const"], model.params["t"], model.params["pred_y"]
        return a1

class State(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_init_params(self):
        model = sm.OLS(self.y, self.X).fit()  # 拟合线性回归模型
        a0, a1, a2 = model.params["const"], model.params["t"], model.params["pred_y"]
        residuals = model.resid  # 计算残差
        residual_var = np.var(residuals)  # 计算残差方差
        degree = 30 # 初始degree
        return a0, a1, a2, residual_var, degree
    
    @staticmethod
    def base_function(degree, target_value):
        return psi(degree/2) - np.log(degree/2) - target_value # 定义目标函数
    
    def calc_eta_lneta(self, a0, a1, a2, residual_var, degree):
        xi = degree/2 + 1/2
        residual = self.y - np.dot([a0, a1, a2], self.X.T)
        zeta = degree/2 + np.power(residual, 2)/(2*residual_var)
        eta = xi / zeta
        ln_eta = psi(xi) - np.log(zeta)  # 计算对应的二伽马函数值
        return eta, ln_eta, xi, zeta, residual
    
    @staticmethod
    def calc_f(residual_var, degree, eta, ln_eta, xi, zeta, residual):
        ln_N = np.sum(-0.5*np.log(2*np.pi*residual_var) + 0.5*ln_eta - eta*np.power(residual, 2)/(2*residual_var))
        ln_G = degree/2*np.log(degree/2) - np.log(gamma(degree/2)) + (degree/2-1)*ln_eta - degree/2*eta
        ln_q = xi*np.log(zeta) - np.log(gamma(xi)) + (xi-1)*ln_eta - xi
        f = np.sum(ln_N + ln_G + ln_q)
        return f

    def calc_a_by_iteration(self, eta, a0, a1, a2):
        # M-step
        v0 = self.y - np.dot([a1, a2], self.X[["t", "pred_y"]].T)
        new_a0 = np.sum(v0*eta) / np.sum(eta)

        v1 = self.y - np.dot([a0, a2], self.X[["const", "pred_y"]].T)
        new_a1 = np.sum(v1*self.X["t"]*eta) / np.sum(eta*np.power(self.X["t"],2))

        v2 = self.y - np.dot([a0, a1], self.X[["const", "t"]].T)
        new_a2 = np.sum(v2*self.X["pred_y"]*eta) / np.sum(eta*np.power(self.X["pred_y"],2))
        return new_a0, new_a1, new_a2

    def calc_a_by_equation(self, eta):
        # M-step
        m0, m1, m2, m3 = np.sum(eta), np.sum(eta*self.y), np.sum(eta*self.X["t"]), np.sum(eta*self.X["pred_y"])
        n0, n1, n2, n3 = np.sum(eta*np.power(self.X["t"],2)), np.sum(eta*self.X["t"]*self.y), np.sum(eta*self.X["t"]), np.sum(eta*self.X["t"]*self.X["pred_y"])
        k0, k1, k2, k3 = np.sum(eta*np.power(self.X["pred_y"],2)), np.sum(eta*self.y*self.X["pred_y"]), np.sum(eta*self.X["pred_y"]), np.sum(eta*self.X["t"]*self.X["pred_y"])
        matrix = [[m0, m2, m3], [n2, n0, n3], [k2, k3, k0]]
        target = [m1, n1, k1]
        solution = np.linalg.solve(matrix, target)
        a0, a1, a2 = solution[0], solution[1], solution[2]
        return a0, a1, a2

    
    def run(self):
        res = []
        f_list = []
        # init E-step
        a0, a1, a2, residual_var, degree = self.get_init_params()
        eta, ln_eta, xi, zeta, residual = self.calc_eta_lneta(a0, a1, a2, residual_var, degree) 
        f = State.calc_f(residual_var, degree, eta, ln_eta, xi, zeta, residual)
        f_list.append(f)
        res.append([0, a0, a1, a2, residual_var, degree, eta.mean(), ln_eta.mean(), xi, zeta.mean(), residual.mean(), 1])
        # print(f"params   init: {a0=}, {a1=}, {a2=}, {residual_var=}, {degree=}")

        for i in range(20):
            # M-step
            # a0, a1, a2 = self.calc_a_by_iteration(eta, a0, a1, a2) # 基于preivous的a0, a1, a2计算当前的a0, a1, a2
            a0, a1, a2 = self.calc_a_by_equation(eta) # 求解方程组计算当前的a0, a1, a2

            # 计算sigma^2
            residual = self.y - np.dot([a0, a1, a2], self.X.T)
            residual_var = np.mean(np.power(residual, 2)*eta)

            # 计算degree
            target_value = np.mean(1 + ln_eta - eta)
            target_function = partial(State.base_function, target_value=target_value)
            solution = fsolve(target_function, degree)
            degree = solution[0]
            # degree = 100000

            # E-step
            eta, ln_eta, xi, zeta, residual = self.calc_eta_lneta(a0, a1, a2, residual_var, degree)
            f = State.calc_f(residual_var, degree, eta, ln_eta, xi, zeta, residual)
            
            # Converge check
            C1 = len(f_list) >= 3
            prev_mean_f = np.mean(f_list[-3:])
            rel_diff = f / prev_mean_f - 1
            C2 = np.abs(rel_diff) <= 0.01
            res.append([i+1, a0, a1, a2, residual_var, degree, eta.mean(), ln_eta.mean(), xi, zeta.mean(), residual.mean(), rel_diff])
            f_list.append(f)
            if C1 and C2:
                break

        res_df = pd.DataFrame(data=res, columns=["round", "a0", "a1", "a2", "residual_var", "degree", "eta", "ln_eta", "xi", "zeta", "residual", "f_rel_diff"])
        p_value = 1
        sig = 0
        return a0, a1, a2, residual_var, degree, p_value, sig, res_df

    def run_with_constant_degree(self, init_degree):
        res = []
        # init E-step
        a0, a1, a2, residual_var, degree = self.get_init_params()
        degree = init_degree
        eta, ln_eta, xi, zeta, residual = self.calc_eta_lneta(a0, a1, a2, residual_var, degree) 
        res.append([0, a0, a1, a2, residual_var, degree, eta.mean(), ln_eta.mean(), xi, zeta.mean(), residual.mean()])
        params = [a0, a1, a2]
        # for i in tqdm(range(1000)):
        for i in range(1000):
            # M-step
            # a0, a1, a2 = self.calc_a_by_iteration(eta, a0, a1, a2) # 基于preivous的a0, a1, a2计算当前的a0, a1, a2
            a0, a1, a2 = self.calc_a_by_equation(eta) # 求解方程组计算当前的a0, a1, a2

            # 计算sigma^2
            residual = self.y - np.dot([a0, a1, a2], self.X.T)
            residual_var = np.mean(np.power(residual, 2)*eta)

            # E-step
            eta, ln_eta, xi, zeta, residual = self.calc_eta_lneta(a0, a1, a2, residual_var, degree)
            res.append([i+1, a0, a1, a2, residual_var, degree, eta.mean(), ln_eta.mean(), xi, zeta.mean(), residual.mean()])
            
            # Converge check
            if np.allclose(params, [a0, a1, a2], atol=1e-3):
                # print(f'{i}th iteration STATE:{a1:.5f}')
                break
            
            params = [a0, a1, a2]
            

        res_df = pd.DataFrame(data=res, columns=["round", "a0", "a1", "a2", "residual_var", "degree", "eta", "ln_eta", "xi", "zeta", "residual"])
        p_value = 1
        sig = 0
        return a0, a1, a2, residual_var, degree, p_value, sig, res_df


def run_one_experiment(X, y, random_state, debug=False):
    """
    This function performs an experiment to compare the effect size between two groups using 
    DIM, CUPED, and State algorithms.

    Parameters:
    X (DataFrame): A DataFrame containing the independent variables (t, pred_y).
    y (Series): A Series containing the dependent variable (post_y).
    random_state (int): The random seed for reproducibility.

    Returns:
    res (list): A list of dictionaries containing the results of the experiment for each algorithm.
    """
    N = len(y)

    # DIM
    dim = Dim(X, y)
    dim_a1, mean_A, mean_B, var, p_value, sig = dim.run()
    dim_rel_diff = dim_a1 / mean_A
    dim_res = dict(random_state=random_state,
                   method="1_DIM",
                   n=N,
                   mean_A=mean_A,
                   abs_diff=dim_a1,
                   rel_diff=dim_rel_diff,
                   var=var,
                   var_reduce=0,
                   p_value=p_value,
                   sig=sig)

    # CUPED
    cuped = Cuped(X, y)
    cuped_a1, cuped_var, p_value, sig = cuped.run()
    cuped_rel_diff = cuped_a1 / mean_A
    cuped_y_var_reduce = cuped_var / var - 1
    cuped_res = dict(random_state=random_state,
                   method="2_CUPED",
                   n=N,
                   mean_A=mean_A,
                   abs_diff=cuped_a1,
                   rel_diff=cuped_rel_diff,
                   var=cuped_var,
                   var_reduce=cuped_y_var_reduce,
                   p_value=p_value,
                   sig=sig)

    # State
    state = State(X, y)
    state_a0, state_a1, state_a2, residual_var, degree, p_value, sig, res_df = state.run()
    # state_a0, state_a1, state_a2, residual_var, degree, p_value, sig, res_df = state.run_with_constant_degree(init_degree=10000000)
    if debug:
        print(res_df)
    state_rel_diff = state_a1 / mean_A
    state_y_var_reduce = residual_var / var - 1
    state_res = dict(random_state=random_state,
                   method="3_STATE",
                   n=N,
                   mean_A=mean_A,
                   abs_diff=state_a1,
                   rel_diff=state_rel_diff,
                   var=np.nan,
                   var_reduce=np.nan,
                   p_value=np.nan,
                   sig=np.nan)
    res = [dim_res, cuped_res, state_res]
    return res
