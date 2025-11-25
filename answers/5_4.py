"""
題目 5.4: 非正半定矩陣的多變量常態分佈模擬（先進行 Higham 修正）
(Multivariate Normal Simulation with Non-PSD Matrix - Higham correction first)

問題描述：
處理非正半定的共變異數矩陣，使用 Higham 方法修正為正半定矩陣，
然後進行多變量常態分佈模擬。比較 Higham 方法與 Near-PSD 方法
在矩陣修正上的差異。

目標：
1. 載入非正半定的共變異數矩陣
2. 使用 Higham 方法修正矩陣
3. 使用修正後的矩陣進行多變量常態分佈模擬
4. 驗證模擬結果並與 Near-PSD 方法比較

解法流程：
1. 讀取 test5_3.csv 檔案中的非正半定矩陣
2. 使用 Utils.higham_covariance() 修正矩陣
3. 使用修正後的矩陣進行多變量常態分佈模擬
4. 設定隨機種子為4以確保結果可重現
5. 計算生成樣本的共變異數矩陣
6. 輸出結果矩陣
"""

import pandas as pd
import numpy as np
import library as Utils

# Test 5.4: Multivariate Normal Simulation with Non-PSD Matrix (Higham correction first)
cin = pd.read_csv("../testfiles/data/test5_3.csv", header=None, skiprows=1).values.astype(float)

# Step 1: Fix non-PSD matrix using Higham method
cin_df = pd.DataFrame(cin)
fixed_matrix = Utils.higham_covariance(cin_df)

# Step 2: Generate samples using the corrected matrix
sampled_data = Utils.simulate_multivariate_normal(
    mean=np.zeros(fixed_matrix.shape[0]),
    cov=fixed_matrix.values,
    n_samples=100000,
    seed=4
)

# Calculate sample covariance matrix
result_matrix = Utils.calculate_cov(sampled_data)
result = pd.DataFrame(result_matrix)
print(result)