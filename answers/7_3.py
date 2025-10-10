"""
作業目標 7.3: t分布迴歸模型 (t-Distribution Regression)

背景:
傳統OLS迴歸假設誤差項遵循常態分布，但金融數據的誤差項
往往呈現厚尾特性，使用t分布誤差項的迴歸模型更為適當。

問題:
如何建立以t分布為誤差項的迴歸模型？

解法 - t迴歸模型:
模型形式：y = Xβ + ε，其中 ε ~ t(ν, 0, σ)

參數估計:
使用最大概似估計法同時估計：
- β: 迴歸係數向量
- σ: 誤差項尺度參數  
- ν: 自由度參數

概似函數:
L(β,σ,ν) = ∏ᵢ f_t(yᵢ - xᵢ'β; ν, 0, σ)

估計步驟:
1. OLS估計提供初始值
2. 計算殘差標準差作為σ初始值
3. 設定ν初始值（通常5-10）
4. 數值優化最大化概似函數

優點比較:
相對於OLS迴歸：
- 對異常值更穩健
- 更準確的參數標準誤
- 更適合金融數據特性

相對於常態迴歸：
- 捕捉厚尾分布特性
- 更保守的信賴區間
- 更準確的預測區間

技術實現:
- 約束優化確保σ>0, ν>2
- L-BFGS-B算法處理邊界約束
- 良好初始值避免局部最優

應用場景:
- 風險因子模型
- 資產定價迴歸
- 總體經濟變數關係
- 異常值較多的數據

模型診斷:
- 殘差分析
- QQ圖檢驗
- 參數穩定性測試

實務價值:
在金融計量中，t迴歸提供更現實的誤差項假設。
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize

if __name__ == "__main__":
    # 讀取迴歸數據
    cin = pd.read_csv("../testfiles/data/test7_3.csv")
    y = cin["y"].values.astype(float)
    X = cin.drop(columns=["y"]).values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])  # 添加截距項
    
    # 定義負對數概似函數
    def nll(theta):
        beta = theta[:-2]    # 迴歸係數
        sigma = theta[-2]    # 尺度參數
        nu = theta[-1]       # 自由度參數
        
        # 參數約束：σ>0, ν>2（確保方差存在）
        if sigma <= 0 or nu <= 2:
            return np.inf
            
        # 計算殘差
        r = y - X @ beta
        
        # 計算t分布的對數概似函數
        ll = stats.t.logpdf(r, df=nu, loc=0.0, scale=sigma).sum()
        return -ll
    
    # 設定初始值
    # 使用OLS估計作為β的初始值
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_ols
    sigma0 = np.std(resid, ddof=X.shape[1])  # 殘差標準差
    nu0 = 5.0  # 自由度初始值
    theta0 = np.concatenate([beta_ols, [max(sigma0, 1e-3), nu0]])
    
    # 設定參數邊界約束
    p = X.shape[1]
    bounds = [(None, None)] * p + [(1e-8, None), (2.0001, None)]
    
    # 執行最大概似估計
    res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)
    
    # 提取估計結果
    beta_hat = res.x[:-2]    # 迴歸係數
    sigma_hat = res.x[-2]    # 尺度參數
    nu_hat = res.x[-1]       # 自由度參數
    mu_hat = 0.0             # t分布中心化
    
    # 整理結果
    result = pd.DataFrame({
        "mu":    [mu_hat],
        "sigma": [sigma_hat],
        "nu":    [nu_hat],
        "Alpha": [beta_hat[0]],  # 截距項
        "B1":    [beta_hat[1]],  # 第一個解釋變數係數
        "B2":    [beta_hat[2]],  # 第二個解釋變數係數
        "B3":    [beta_hat[3]],  # 第三個解釋變數係數
    })
    
    print(result)