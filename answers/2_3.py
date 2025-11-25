"""
題目 2.3: 混合指數移動平均共變異數矩陣 (Mixed EWMA Covariance Matrix)

問題描述：
建立混合指數移動平均共變異數矩陣，結合不同衰減參數的優點。
使用較長記憶的 λ=0.97 來計算相關係數（捕捉長期關聯性），
使用較短記憶的 λ=0.94 來計算變異數（更快響應波動性變化）。

目標：
1. 載入時間序列資料
2. 分別使用兩個不同的 λ 值計算指數加權矩陣
3. 從 λ=0.97 提取相關係數矩陣
4. 從 λ=0.94 提取標準差向量
5. 結合兩者建立混合共變異數矩陣

解法流程：
1. 讀取 test2.csv 資料
2. 使用 λ=0.97 計算長期共變異數和相關係數矩陣
3. 使用 λ=0.94 計算短期共變異數矩陣
4. 從各自的共變異數矩陣提取標準差
5. 使用公式 Σ = D(std_94) * corr_97 * D(std_94) 組合結果
6. 輸出混合共變異數矩陣
"""

import pandas as pd
import numpy as np
import library as Utils

# Test 2.3: Mixed EWMA Covariance Matrix
data = pd.read_csv("../testfiles/data/test2.csv")

# Step 1: Calculate correlation matrix with λ=0.97 (long memory)
cov_97, corr_97 = Utils.ew_cov_corr_normalized(data, lam=0.97)
std_97 = np.sqrt(np.diag(cov_97))

# Step 2: Calculate covariance matrix with λ=0.94 (short memory for variances)
cov_94, corr_94 = Utils.ew_cov_corr_normalized(data, lam=0.94)
std_94 = np.sqrt(np.diag(cov_94))

# Step 3: Combine - use correlation from λ=0.97 with variances from λ=0.94
# Formula: Σ = D(std_94) * corr_97 * D(std_94)
result_matrix = Utils._to_cov_from_corr(corr_97.values, std_94)

# Convert to DataFrame with column names
result = pd.DataFrame(result_matrix, columns=data.columns, index=data.columns)
print(result)