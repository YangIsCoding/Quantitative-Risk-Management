"""
題目 8.6: 使用蒙地卡羅模擬的預期損失計算 (Expected Shortfall using Monte Carlo Simulation)

問題描述：
使用蒙地卡羅模擬結合配適的 t 分佈來計算預期損失（ES）。蒙地卡羅方法
提供了一個靈活的框架，可以處理複雜的分佈和風險因子，是現代風險管理
中的重要工具。

目標：
1. 載入報酬率資料並配適 t 分佈
2. 使用蒙地卡羅模擬生成大量樣本
3. 從模擬樣本計算 5% 信心水準的 ES
4. 驗證模擬結果與解析方法的一致性

解法流程：
1. 讀取 test7_2.csv 檔案中的報酬率資料
2. 提取 x1 欄位作為報酬率時間序列
3. 調用 Utils.es_sim_from_fitted_t() 進行蒙地卡羅 ES 計算
4. 設定 α=0.05、模擬次數=100,000、隨機種子=42
5. 輸出 ES 計算結果
"""

import pandas as pd
import library as Utils

# Test 8.6: Expected Shortfall using Monte Carlo Simulation
data = pd.read_csv("../testfiles/data/test7_2.csv")
returns = data["x1"]

# Calculate Expected Shortfall using Monte Carlo simulation with t-distribution
result_dict = Utils.es_sim_from_fitted_t(returns, alpha=0.05, n_sim=100000, random_state=42)
result = pd.DataFrame([result_dict])
print(result)