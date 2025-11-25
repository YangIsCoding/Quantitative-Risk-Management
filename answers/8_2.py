"""
題目 8.2: t 分佈 VaR 計算 (t-Distribution VaR Calculation)

問題描述：
使用 t 分佈假設計算風險值（VaR）。相較於常態分佈，t 分佈具有更厚的尾部，
更適合描述金融市場的極端事件，通常會給出更保守（更大）的 VaR 估計值。

目標：
1. 載入報酬率資料
2. 假設報酬率服從 t 分佈
3. 計算 5% 信心水準的 VaR
4. 比較與常態分佈 VaR 的差異

解法流程：
1. 讀取 test7_2.csv 檔案中的報酬率資料
2. 提取 x1 欄位作為報酬率時間序列
3. 調用 Utils.var_from_returns() 以 t 分佈計算 VaR
4. 設定 α=0.05（95% 信心水準）
5. 輸出 VaR 計算結果
"""

import pandas as pd
import library as Utils

# Test 8.2: t-Distribution VaR Calculation
data = pd.read_csv("../testfiles/data/test7_2.csv")
returns = data["x1"]

# Calculate VaR using t-distribution assumption
result_dict = Utils.var_from_returns(returns, alpha=0.05, dist="t")
result = pd.DataFrame([result_dict])
print(result)