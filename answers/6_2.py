"""
題目 6.2: 對數報酬率計算 (Log Returns Calculation)

問題描述：
從資產價格資料計算對數報酬率（連續複利報酬率）。對數報酬率的計算公式為：
rt = ln(Pt/Pt-1) = ln(Pt) - ln(Pt-1)。對數報酬率具有時間可加性的優點，
在量化金融中廣泛使用。

目標：
1. 載入包含價格資料的檔案
2. 計算每個資產的對數報酬率
3. 處理第一期的缺失值（因為沒有前期價格）
4. 輸出對數報酬率資料

解法流程：
1. 讀取 test6.csv 價格資料
2. 使用對數函數計算 ln(Pt/Pt-1)
3. 移除第一行的 NaN 值
4. 顯示前幾行資料作為驗證
"""

import pandas as pd
import numpy as np

# Test 6.2: Log Returns Calculation
prices = pd.read_csv("../testfiles/data/test6.csv")

# Calculate log returns: rt = ln(Pt/Pt-1) = ln(Pt) - ln(Pt-1)
# Assuming first column is date and rest are prices
returns = prices.copy()
for col in prices.columns:
    if col != "Date":
        returns[col] = np.log(prices[col] / prices[col].shift(1))

# Remove first row (NaN values)
returns = returns.dropna()
print(returns.head())