"""
題目 5.5: 主成分分析模擬 (PCA Simulation)

問題描述：
使用主成分分析（PCA）降維技術處理高維共變異數矩陣，然後進行
多變量常態分佈模擬。PCA 可以減少計算複雜度並處理接近奇異的矩陣，
同時保留大部分變異性。

目標：
1. 載入高維共變異數矩陣
2. 使用 PCA 降維至保留 99% 變異性的維度
3. 使用降維後的矩陣進行多變量常態分佈模擬
4. 驗證 PCA 降維的效果

解法流程：
1. 讀取 test5_2.csv 檔案中的共變異數矩陣
2. 使用 Utils.pca_covariance() 進行 PCA 降維
3. 設定保留 99% 變異性的閾值
4. 使用降維後的共變異數矩陣進行模擬
5. 設定隨機種子為4以確保結果可重現
6. 計算生成樣本的共變異數矩陣
7. 輸出結果矩陣
"""

import pandas as pd
import numpy as np
import library as Utils

# Test 5.5: PCA Simulation
cin = pd.read_csv("../testfiles/data/test5_2.csv", header=None, skiprows=1).values.astype(float)

# Use PCA to reduce dimensionality and simulate
cin_df = pd.DataFrame(cin)
cov_pca, n_components, explained_ratios = Utils.pca_covariance(cin_df, threshold=0.99)

# Generate samples using the PCA-reduced covariance matrix
sampled_data = Utils.simulate_multivariate_normal(
    mean=np.zeros(cov_pca.shape[0]),
    cov=cov_pca.values,
    n_samples=100000,
    seed=4
)

# Calculate sample covariance matrix
result_matrix = Utils.calculate_cov(sampled_data)
result = pd.DataFrame(result_matrix)
print(result)