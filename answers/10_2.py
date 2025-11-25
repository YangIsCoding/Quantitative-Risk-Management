"""
題目 10.2: 自訂風險預算的風險平價投資組合 (Risk Parity Portfolio with Custom Risk Budget)

問題描述：
建立具有自訂風險預算的風險平價投資組合。與等風險預算不同，
這裡可以為不同資產分配不同的目標風險貢獻，提供更靈活的
風險分配策略。

目標：
1. 載入資產共變異數矩陣
2. 設定自訂風險預算 [2,2,2,2,1]
3. 計算滿足目標風險預算的投資組合權重
4. 確保每個資產的風險貢獻與其預算成正比

解法流程：
1. 讀取 test5_2.csv 檔案中的共變異數矩陣
2. 定義自訂風險預算向量 [2,2,2,2,1]
3. 使用 Utils.risk_parity_csd() 計算權重
4. 解決約束優化問題以達成目標風險預算
5. 輸出投資組合權重向量
"""

import library as Utils
import pandas as pd
import numpy as np

# Test10.2: Risk Parity Portfolio with Custom Risk Budget
cov = pd.read_csv("../testfiles/data/test5_2.csv").values

# Custom risk budget: [2, 2, 2, 2, 1]
budget = np.array([2, 2, 2, 2, 1])

# Using the risk_parity_csd function from library with custom budget
weights = Utils.risk_parity_csd(cov, budget)
print(weights)