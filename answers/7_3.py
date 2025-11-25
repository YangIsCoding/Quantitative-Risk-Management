"""
題目 7.3: t 分佈回歸分析 (t-Distribution Regression)

問題描述：
使用 t 分佈進行回歸分析，相較於傳統的常態分佈線性回歸，
t 分佈回歸對異常值更具強健性，更適合處理金融資料中的厚尾現象。
使用最大概似估計法同時估計回歸參數和分佈參數。

目標：
1. 載入回歸資料（因變數 y 和自變數 X）
2. 建立 t 分佈回歸模型
3. 使用 MLE 估計回歸係數和分佈參數
4. 輸出截距、斜率係數及分佈參數

解法流程：
1. 讀取 test7_3.csv 檔案，分離 y 和 X 變數
2. 添加截距項到設計矩陣
3. 定義負對數概似函數
4. 使用 OLS 提供初始參數估計
5. 使用 L-BFGS-B 優化算法進行 MLE
6. 輸出估計的回歸係數（Alpha, B1, B2, B3）和分佈參數（μ, σ, ν）
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize

# Test 7.3: t-Distribution Regression
cin = pd.read_csv("../testfiles/data/test7_3.csv")
y = cin["y"].values.astype(float)
X = cin.drop(columns=["y"]).values.astype(float)
X = np.column_stack([np.ones(len(X)), X])  # Add intercept

# Define negative log-likelihood function
def nll(theta):
    beta = theta[:-2]
    sigma = theta[-2]
    nu = theta[-1]
    
    if sigma <= 0 or nu <= 2:
        return np.inf
        
    r = y - X @ beta
    ll = stats.t.logpdf(r, df=nu, loc=0.0, scale=sigma).sum()
    return -ll

# Initial values using OLS
beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta_ols
sigma0 = np.std(resid, ddof=X.shape[1])
nu0 = 5.0
theta0 = np.concatenate([beta_ols, [max(sigma0, 1e-3), nu0]])

# Parameter bounds
p = X.shape[1]
bounds = [(None, None)] * p + [(1e-8, None), (2.0001, None)]

# Maximum likelihood estimation
res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)

# Extract results
beta_hat = res.x[:-2]
sigma_hat = res.x[-2]
nu_hat = res.x[-1]
mu_hat = 0.0

# Prepare result
result = pd.DataFrame({
    "mu":    [mu_hat],
    "sigma": [sigma_hat],
    "nu":    [nu_hat],
    "Alpha": [beta_hat[0]],
    "B1":    [beta_hat[1]],
    "B2":    [beta_hat[2]],
    "B3":    [beta_hat[3]],
})

print(result)