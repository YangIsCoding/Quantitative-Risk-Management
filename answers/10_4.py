"""
題目 10.4: 有界限約束的最大夏普比率投資組合 (Maximum Sharpe Ratio Portfolio with Bounds Constraints)

問題描述：
在權重界限約束下建立最大化夏普比率的投資組合。實際投資中常有
投資限制，例如每個資產的最低和最高投資比例，需要在這些約束條件下
尋找最優投資組合。

目標：
1. 載入資產共變異數矩陣和期望報酬率
2. 設定權重界限約束 [0.1, 0.5]
3. 在約束條件下找到最大夏普比率投資組合
4. 輸出滿足約束的最優權重

解法流程：
1. 讀取 test5_2.csv 檔案中的共變異數矩陣
2. 讀取 test10_3_means.csv 檔案中的期望報酬率
3. 設定無風險利率 rf=0.04 和權重界限 [0.1, 0.5]
4. 使用 Utils.max_sharpe_ratio() 進行約束優化
5. 設定 long_only=True 確保只做多頭
6. 輸出滿足約束的最優投資組合權重
"""

import library as Utils
import pandas as pd

# Test10.4: Maximum Sharpe Ratio Portfolio with Bounds Constraints
cov = pd.read_csv("../testfiles/data/test5_2.csv").values
mu = pd.read_csv("../testfiles/data/test10_3_means.csv").values.flatten()

# Maximum Sharpe ratio optimization with rf = 0.04 and bounds [0.1, 0.5] for all assets
# Using the max_sharpe_ratio function from library with bounds constraints
weights, sharpe_ratio = Utils.max_sharpe_ratio(mu, cov, rf=0.04, bounds=[(0.1, 0.5)] * 5, long_only=True)
print(weights)