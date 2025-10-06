import pandas as pd
import numpy as np
import sys
sys.path.append('..')

# 讀取共變異數矩陣（test5_2 是 PSD 的矩陣）
cov_matrix = pd.read_csv("../testfiles/data/test5_2.csv")

# 移除第一列（列名），只保留數值
cov_values = cov_matrix.values

# 實作 PCA 模擬 (pctExp=0.99)
# PCA: 進行特徵值分解
eigenvals, eigenvecs = np.linalg.eigh(cov_values)

# 按特徵值遞減排序
idx = np.argsort(eigenvals)[::-1]
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

# 找到累積解釋方差達到99%的組件數
cumulative_explained = np.cumsum(eigenvals) / np.sum(eigenvals)
n_components = np.argmax(cumulative_explained >= 0.99) + 1

# 保留前 n_components 個主成分
selected_eigenvals = eigenvals[:n_components]
selected_eigenvecs = eigenvecs[:, :n_components]

# 使用PCA進行模擬 (100000個樣本)
np.random.seed(4)
# 產生標準正態分佈的隨機數
Z = np.random.randn(100000, n_components)

# 乘以特徵值的平方根來獲得適當的方差
scaled_Z = Z * np.sqrt(selected_eigenvals)

# 轉換回原始空間
simulated_data = scaled_Z @ selected_eigenvecs.T

# 計算模擬資料的共變異數矩陣
result_matrix = np.cov(simulated_data.T)

# 轉換為 DataFrame 格式，移除第一列
result_df = pd.DataFrame(result_matrix[:, 1:], 
                        index=['x1', 'x2', 'x3', 'x4', 'x5'],
                        columns=['x2', 'x3', 'x4', 'x5'])

print(result_df)