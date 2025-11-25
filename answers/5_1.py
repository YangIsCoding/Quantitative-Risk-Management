"""
題目 5.1: 使用正半定共變異數矩陣進行多變量常態分佈模擬 
(Multivariate Normal Simulation with PSD Covariance Matrix)

問題描述：
使用給定的正半定共變異數矩陣生成多變量常態分佈的隨機樣本。
這是蒙地卡羅模擬在風險管理中的基礎應用，用於模擬多個相關資產
的價格變動。

目標：
1. 載入正半定共變異數矩陣
2. 設定均值向量為零向量
3. 生成大量（100,000個）多變量常態分佈樣本
4. 計算樣本的共變異數矩陣以驗證模擬品質

解法流程：
1. 讀取 test5_1.csv 檔案中的共變異數矩陣
2. 使用 Utils.simulate_multivariate_normal() 生成樣本
3. 設定隨機種子為4以確保結果可重現
4. 計算生成樣本的實際共變異數矩陣
5. 輸出結果矩陣以驗證與理論值的接近程度
"""

import pandas as pd
import numpy as np
import library as Utils

# Test 5.1: Multivariate Normal Simulation with PSD Covariance Matrix
cin = pd.read_csv("../testfiles/data/test5_1.csv", header=None, skiprows=1).values.astype(float)

# Generate samples using the simulate_multivariate_normal function
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