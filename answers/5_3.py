import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import near_psd, chol_psd

# 讀取共變異數矩陣（test5_3是非PSD的矩陣）
cov_matrix = pd.read_csv("../testfiles/data/test5_3.csv")

# 移除第一列（列名），只保留數值
cov_values = cov_matrix.values

# 需要實作 simulateNormal 函數來模擬，並計算共變異數
# 根據 test_setup.jl：simulateNormal(100000, cin,fixMethod=near_psd)
# 先用 near_psd 修正矩陣，然後模擬

# 應用 near_psd 函數修正矩陣
fixed_matrix = near_psd(cov_values)

# 手動處理 PSD 並進行 Cholesky 分解
eigenvals, eigenvecs = np.linalg.eigh(fixed_matrix)
eigenvals = np.maximum(eigenvals, 1e-8)  # 確保所有特徵值為正
fixed_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

# 模擬正態分佈（100000個樣本）
np.random.seed(4)  # 根據 test_setup.jl 設定的種子
L = np.linalg.cholesky(fixed_matrix)
Z = np.random.randn(100000, 5)
simulated_data = Z @ L.T

# 計算模擬資料的共變異數矩陣
result_matrix = np.cov(simulated_data.T)

# 轉換為 DataFrame 格式，移除第一列
result_df = pd.DataFrame(result_matrix[:, 1:], 
                        index=['x1', 'x2', 'x3', 'x4', 'x5'],
                        columns=['x2', 'x3', 'x4', 'x5'])

print(result_df)