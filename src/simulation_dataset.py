import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
import os, sys

os.chdir(sys.path[0])

# 定义非线性函数 b(Xi) 和 c(Xi)
def b(X):
    return (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 
            6 * X[:, 2]**2 + 
            10 * np.abs(X[:, 3]) + 
            5 * np.abs(X[:, 4]) + 
            50)

def c(X):
    return (10 * np.sin(np.pi * X[:, 3]) * X[:, 4] + 
            15 * (X[:, 1] + X[:, 2])**2 + 
            5 * np.abs(X[:, 0]) + 
            30)

# 定义处理效应 τy(Xi) 和 τz(Xi)
def tau_y(X):
    return (X[:, 0] * X[:, 2] + 
            np.log(1 + np.exp(X[:, 1])))


def conditional_tau_y(X):
    '''
    若 X4 > 90分位数, 返回 X1 * X2 + log(1 + exp(X3))，否则返回 0
    '''
    result = 5 * X[:, 0] * X[:, 2] + np.log(1 + np.exp(X[:, 1]))
    
    # 使用 np.where 根据条件返回结果
    threshould = np.percentile(X[:, 3], 90)  # 90%分位数
    print(f"Condition: X4 > {threshould}")  # 打印分位数的条件
    condition = X[:, 3] > threshould
    return np.where(condition, result, 0)

def tau_z(X):
    return (X[:, 1]**2 + 
            3 * np.log(1 + np.exp(X[:, 3]) + np.abs(X[:, 4])))

def generate(n, tau_y_type):
    # 从均匀分布 U(0, 10) 中生成均值向量 u
    u = np.random.uniform(0, 10, size=5)
    scale = np.random.uniform(0, 10, size=5)
    for i in range(5):
        print(f"X{i+1} ~ N({u[i]}, {scale[i]})")

    # 生成协变量 Xi，样本数量为n
    X = np.random.normal(loc=u, scale=scale, size=(n, 5))

    # 随机生成处理变量 Ti
    T = np.random.binomial(1, p=0.5, size=n)  # 二项分布，处理组的概率为0.5

    # 定义误差项 εi 和 ηi
    epsilon = np.random.normal(0, 25, size=n)   # εi ~ N(0, 25^2)
    eta = np.random.normal(0, 10, size=n)       # ηi ~ N(0, 10^2)

    # 构建结果变量 Yi 和 Zi
    if tau_y_type == "normal":
        tau_y_res = tau_y(X)
    elif tau_y_type == "conditional":
        tau_y_res = conditional_tau_y(X)
    else:
        raise ValueError("tau_y_type must be 'normal' or 'conditional'")
    Y = b(X) + T * tau_y_res + epsilon
    Z = c(X) + T * tau_z(X) + eta

    # 创建 DataFrame
    data = pd.DataFrame({
        'X1': X[:, 0],
        'X2': X[:, 1],
        'X3': X[:, 2],
        'X4': X[:, 3],
        'X5': X[:, 4],
        'tau_y': tau_y_res,
        'T': T,
        'Y': Y,
        'Z': Z
    })

    # 添加离群值
    sigma_y = Y.std()
    sigma_z = Z.std()
    Y_outliers = np.random.uniform(Y + 4 * sigma_y, Y + 20 * sigma_y)
    Z_outliers = np.random.uniform(Z + 4 * sigma_z, Z + 20 * sigma_z)

    # 随机选择一些样本来替换为离群值
    num_outliers = int(0.05 * len(data)) # 假设5%的离群值
    outlier_indices_Y = np.random.choice(data.index, num_outliers, replace=False)
    outlier_indices_Z = np.random.choice(data.index, num_outliers, replace=False)

    data.loc[outlier_indices_Y, 'Y'] = Y_outliers[:num_outliers]
    data.loc[outlier_indices_Z, 'Z'] = Z_outliers[:num_outliers]

    return data

def add_predict(data):
    # 定义自变量和因变量
    X = data[['X1', 'X2', 'X3', 'X4', 'X5']]
    y = data['Y']
    data['Pred_Y'] = np.nan

    # 设置2-fold交叉验证
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        # 分割训练集和测试集
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 筛选出 T=0 的训练数据集
        train_data_T0 = X_train[data.loc[train_index, 'T'] == 0]
        train_target_T0 = y_train[data.loc[train_index, 'T'] == 0]

        # 创建XGBoost模型
        model = xgb.XGBRegressor(objective='reg:squarederror',
                                booster='gbtree',
                                n_estimators=200,
                                learning_rate=0.01,
                                max_depth=6,
                                min_child_weight=1,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                gamma=0,
                                reg_alpha=0,
                                reg_lambda=1,
                                random_state=42,
                                verbosity=1)
        
        # 训练模型
        model.fit(train_data_T0, train_target_T0)
        preds = model.predict(X_test)
        data.loc[test_index, 'Pred_Y'] = preds

    return data

def generate_simulation_dataset(n, tau_y_type, seed=42):
    np.random.seed(seed)
    data = generate(n, tau_y_type)
    df = add_predict(data)
    return df

if __name__ == '__main__':
    # n = 1000
    n = 5_000_000
    # tau_y_type = "normal"  # "normal=论文中的tau function"
    tau_y_type = "conditional"  # "conditional=部分用户有effect，部分用户无effect"
    df = generate_simulation_dataset(n, tau_y_type)
    print(df.head())
    df.to_csv(f'../data/simulation_data_{n}_{tau_y_type}.csv', index=False)