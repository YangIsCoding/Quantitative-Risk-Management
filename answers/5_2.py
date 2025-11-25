"""
題目 5.2: 處理接近奇異矩陣的多變量常態分佈模擬 
(Multivariate Normal Simulation with PSD Covariance Matrix - handling near-singular)

問題描述：
處理接近奇異（near-singular）的正半定共變異數矩陣進行多變量常態分佈模擬。
接近奇異的矩陣在數值計算上具有挑戰性，需要穩健的演算法來處理
可能的數值不穩定性。

目標：
1. 載入可能接近奇異的正半定共變異數矩陣
2. 使用穩健的演算法進行多變量常態分佈模擬
3. 生成大量樣本以驗證模擬品質
4. 計算樣本共變異數矩陣以檢驗準確性

解法流程：
1. 讀取 test5_2.csv 檔案中的共變異數矩陣
2. 使用 Utils.simulate_multivariate_normal() 自動處理 PSD 矩陣
3. 設定隨機種子為4以確保結果可重現
4. 計算生成樣本的實際共變異數矩陣
5. 輸出結果矩陣
"""

import pandas as pd
import numpy as np
import library as Utils

# Test 5.2: Multivariate Normal Simulation with PSD Covariance Matrix (handling near-singular)
cin = pd.read_csv("../testfiles/data/test5_2.csv", header=None, skiprows=1).values.astype(float)

# Generate samples using the simulate_multivariate_normal function
# The function automatically handles PSD matrices
sampled_data = Utils.simulate_multivariate_normal(
    mean=np.zeros(cin.shape[0]),
    cov=cin,
    n_samples=100000,
    seed=4
)

# Calculate sample covariance matrix
result_matrix = Utils.calculate_cov(sampled_data)
result = pd.DataFrame(result_matrix)
print(result)