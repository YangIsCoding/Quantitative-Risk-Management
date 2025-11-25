"""
題目 9.1: 使用聯結函數的投資組合風險聚合 (Portfolio Risk Aggregation using Copula)

問題描述：
使用聯結函數（Copula）方法來聚合不同資產的風險，建立投資組合的
風險模型。Copula 可以分離邊際分佈和相依結構，允許不同資產使用
不同的邊際分佈，更靈活地建模風險因子間的相依性。

目標：
1. 載入多資產報酬率資料
2. 為不同資產配適不同的邊際分佈
3. 使用 Copula 生成相關的隨機樣本
4. 計算投資組合的 VaR 和 ES

解法流程：
1. 讀取 test9_1_returns.csv 報酬率資料
2. 設定資產當前價格和持有數量
3. 指定邊際分佈類型（資產A：常態分佈，資產B：t分佈）
4. 使用 Utils.generate_copula_samples() 生成 Copula 樣本
5. 使用 Utils.portfolio_var_es_sim() 計算投資組合風險指標
6. 輸出投資組合的 VaR 和 ES
"""

import pandas as pd
import numpy as np
import library as Utils

# Test 9.1: Portfolio Risk Aggregation using Copula
returns_data = pd.read_csv("../testfiles/data/test9_1_returns.csv")

# Portfolio information
prices = np.array([2000.0, 3000.0])  # Current prices for assets A and B
holdings = np.array([1.0, 1.0])     # Holdings (number of shares)

# Generate copula samples with different marginal distributions
n_assets = 2
dist_types = ["normal", "t"]  # Asset A: normal, Asset B: t-distribution
samples, R, params = Utils.generate_copula_samples(
    n_assets=n_assets,
    dist_types=dist_types, 
    data=returns_data.values,
    n_samples=100000,
    random_state=42
)

# Calculate portfolio VaR and ES
result = Utils.portfolio_var_es_sim(prices, holdings, samples, alpha=0.05)
print(result)