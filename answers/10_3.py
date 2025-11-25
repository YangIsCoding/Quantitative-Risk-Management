"""
題目 10.3: 最大夏普比率投資組合 (Maximum Sharpe Ratio Portfolio)

問題描述：
建立最大化夏普比率的投資組合。夏普比率衡量單位風險的超額報酬，
是現代投資組合理論中重要的績效評估指標。最大夏普比率投資組合
代表風險調整後報酬最佳的投資組合。

目標：
1. 載入資產共變異數矩陣和期望報酬率
2. 設定無風險利率
3. 找到最大化夏普比率的投資組合權重
4. 輸出最優權重和對應的夏普比率

解法流程：
1. 讀取 test5_2.csv 檔案中的共變異數矩陣
2. 讀取 test10_3_means.csv 檔案中的期望報酬率
3. 設定無風險利率 rf=0.04
4. 使用 Utils.max_sharpe_ratio() 進行優化
5. 輸出最優投資組合權重
"""

import pandas as pd
import library as Utils

# Test10.3: Maximum Sharpe Ratio Portfolio
cov = pd.read_csv("../testfiles/data/test5_2.csv").values
mu = pd.read_csv("../testfiles/data/test10_3_means.csv").values.flatten()

# Using the max_sharpe_ratio function from library
weights, sharpe_ratio = Utils.max_sharpe_ratio(mu=mu, cov=cov, rf=0.04)
print(weights)