"""
題目 1.3: 成對方法計算共變異數矩陣 (Pairwise Method for Covariance Matrix)

問題描述：
使用成對方法來計算共變異數矩陣。與跳過缺失值方法不同，成對方法會為每對變數
利用所有可用的觀察值來計算共變異數，不會因為某個觀察值在其他變數中有缺失值
就完全捨棄該觀察值。

目標：
1. 載入包含缺失值的資料
2. 使用成對方法計算共變異數矩陣
3. 利用每對變數的所有可用觀察值
4. 輸出完整的共變異數矩陣

解法流程：
1. 讀取 test1.csv 資料
2. 調用 Utils.pairwise_cov_corr() 以成對方式計算共變異數矩陣
3. 直接輸出結果（函數已返回 DataFrame 格式）
"""

import pandas as pd
import library as Utils

# Test 1.3: Pairwise Method for Covariance Matrix
data = pd.read_csv("../testfiles/data/test1.csv")

# Pairwise method: Calculate covariance for each pair using all available data
cov_matrix, _ = Utils.pairwise_cov_corr(data)

# Convert to DataFrame (already has column names from pairwise_cov_corr)
print(cov_matrix)