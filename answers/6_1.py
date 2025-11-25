"""
題目 6.1: 離散（算術）報酬率計算 (Discrete (Arithmetic) Returns Calculation)

問題描述：
從資產價格資料計算離散報酬率（算術報酬率）。離散報酬率的計算公式為：
rt = (Pt - Pt-1) / Pt-1，這是金融分析中最常用的報酬率計算方式。

目標：
1. 載入包含價格資料的檔案
2. 計算每個資產的離散報酬率
3. 處理第一期的缺失值（因為沒有前期價格）
4. 輸出並保存報酬率資料

解法流程：
1. 讀取 test6.csv 價格資料
2. 使用 pct_change() 函數計算百分比變化
3. 移除第一行的 NaN 值
4. 將結果保存到 testout6_1.csv
5. 顯示前幾行資料作為驗證
"""

import pandas as pd
import numpy as np

# Test 6.1: Discrete (Arithmetic) Returns Calculation
prices = pd.read_csv("../testfiles/data/test6.csv")

# Calculate discrete returns: rt = (Pt - Pt-1) / Pt-1
# Assuming first column is date and rest are prices
returns = prices.copy()
for col in prices.columns:
    if col != "Date":
        returns[col] = prices[col].pct_change()

# Remove first row (NaN values) and save result
returns = returns.dropna()
returns.to_csv("../testfiles/data/testout6_1.csv", index=False)
print(returns.head())