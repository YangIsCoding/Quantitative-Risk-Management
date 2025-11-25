"""
題目 1.1: 跳過缺失值方法計算共變異數矩陣 (Skip Missing Method for Covariance Matrix)

問題描述：
使用跳過缺失值的方法來計算共變異數矩陣。當資料中有缺失值（NaN）時，
需要先移除包含缺失值的觀察值（整行），然後計算剩餘資料的共變異數矩陣。

目標：
1. 載入包含缺失值的資料
2. 移除包含任何缺失值的資料行
3. 計算清理後資料的共變異數矩陣
4. 以 DataFrame 格式輸出結果

解法流程：
1. 讀取 test1.csv 資料
2. 使用 dropna() 移除包含 NaN 的行
3. 調用 Utils.calculate_cov() 計算共變異數矩陣
4. 將結果轉換為帶有行列標籤的 DataFrame
"""

import pandas as pd
import library as Utils

# Test 1.1: Skip Missing Method for Covariance Matrix
data = pd.read_csv("../testfiles/data/test1.csv")

# Skip Missing: Remove rows with any NaN values, then calculate covariance
data_clean = data.dropna()
cov_matrix = Utils.calculate_cov(data_clean.values)

# Convert to DataFrame with column names
result = pd.DataFrame(cov_matrix, columns=data.columns, index=data.columns)
print(result)