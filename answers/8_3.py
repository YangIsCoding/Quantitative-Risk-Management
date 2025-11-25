"""
題目 8.3: 使用 t 分佈的蒙地卡羅 VaR 計算 (Monte Carlo VaR using t-Distribution)

問題描述：
使用蒙地卡羅模擬結合 t 分佈來計算風險值（VaR）。蒙地卡羅方法通過
大量隨機模擬來估計 VaR，相較於解析方法更靈活，能處理複雜的分佈
和投資組合結構。

目標：
1. 載入報酬率資料並估計 t 分佈參數
2. 使用蒙地卡羅模擬生成大量 t 分佈樣本
3. 從模擬樣本計算 5% 信心水準的 VaR
4. 驗證蒙地卡羅結果與解析方法的一致性

解法流程：
1. 讀取 test7_2.csv 檔案中的報酬率資料
2. 提取 x1 欄位作為報酬率時間序列
3. 調用 Utils.var_mc_t_from_returns() 進行蒙地卡羅 VaR 計算
4. 設定 α=0.05、模擬次數=100,000、隨機種子=42
5. 輸出 VaR 計算結果
"""

import pandas as pd
import library as Utils

# Test 8.3: Monte Carlo VaR using t-Distribution
data = pd.read_csv("../testfiles/data/test7_2.csv")
returns = data["x1"]

# Calculate VaR using Monte Carlo simulation with t-distribution
result_dict = Utils.var_mc_t_from_returns(returns, alpha=0.05, n_samples=100000, seed=42)
result = pd.DataFrame([result_dict])
print(result)