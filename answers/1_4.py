"""
題目 1.4: 成對方法計算相關係數矩陣 (Pairwise Method for Correlation Matrix)

問題描述：
使用成對方法來計算相關係數矩陣。成對方法會為每對變數利用所有可用的觀察值
來計算相關係數，這樣可以最大化使用可用資料，避免因為部分缺失值而丟失
有用的資訊。

目標：
1. 載入包含缺失值的資料
2. 使用成對方法計算相關係數矩陣
3. 利用每對變數的所有可用觀察值
4. 輸出完整的相關係數矩陣

解法流程：
1. 讀取 test1.csv 資料
2. 調用 Utils.pairwise_cov_corr() 以成對方式計算相關係數矩陣
3. 直接輸出相關係數矩陣結果
"""

import pandas as pd
import library as Utils

# Test 1.4: Pairwise Method for Correlation Matrix
data = pd.read_csv("../testfiles/data/test1.csv")

# Pairwise method: Calculate correlation for each pair using all available data
_, corr_matrix = Utils.pairwise_cov_corr(data)

# Convert to DataFrame (already has column names from pairwise_cov_corr)
print(corr_matrix)