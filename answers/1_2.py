"""
題目 1.2: 跳過缺失值方法計算相關係數矩陣 (Skip Missing Method for Correlation Matrix)

問題描述：
使用跳過缺失值的方法來計算相關係數矩陣。先移除包含缺失值的觀察值，
然後計算共變異數矩陣，最後將其轉換為相關係數矩陣。

目標：
1. 載入包含缺失值的資料
2. 移除包含任何缺失值的資料行
3. 計算清理後資料的共變異數矩陣
4. 將共變異數矩陣轉換為相關係數矩陣
5. 以 DataFrame 格式輸出結果

解法流程：
1. 讀取 test1.csv 資料
2. 使用 dropna() 移除包含 NaN 的行
3. 調用 Utils.calculate_cov() 計算共變異數矩陣
4. 使用 Utils._to_corr_from_cov() 將共變異數矩陣轉換為相關係數矩陣
5. 將結果轉換為帶有行列標籤的 DataFrame
"""

import pandas as pd
import numpy as np
import library as Utils

# Test 1.2: Skip Missing Method for Correlation Matrix
data = pd.read_csv("../testfiles/data/test1.csv")

# Skip Missing: Remove rows with any NaN values, then calculate correlation
data_clean = data.dropna()
cov_matrix = Utils.calculate_cov(data_clean.values)

# Convert covariance to correlation matrix
corr_matrix, _ = Utils._to_corr_from_cov(cov_matrix)

# Convert to DataFrame with column names
result = pd.DataFrame(corr_matrix, columns=data.columns, index=data.columns)
print(result)