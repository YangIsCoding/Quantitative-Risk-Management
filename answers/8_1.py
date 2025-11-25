"""
題目 8.1: 常態分佈 VaR 計算 (Normal VaR Calculation)

問題描述：
使用常態分佈假設計算風險值（Value at Risk, VaR）。VaR 是風險管理中
重要的風險度量指標，表示在特定信心水準下，投資組合在特定時間內
可能遭受的最大損失。

目標：
1. 載入報酬率資料
2. 假設報酬率服從常態分佈
3. 計算 5% 信心水準的 VaR
4. 輸出 VaR 值和相關統計量

解法流程：
1. 讀取 test7_1.csv 檔案中的報酬率資料
2. 提取 x1 欄位作為報酬率時間序列
3. 調用 Utils.var_from_returns() 以常態分佈計算 VaR
4. 設定 α=0.05（95% 信心水準）
5. 輸出 VaR 計算結果
"""

import pandas as pd
import library as Utils

# Test 8.1: Normal VaR Calculation
data = pd.read_csv('../testfiles/data/test7_1.csv')
returns = data['x1']

# Calculate VaR using normal distribution assumption
result_dict = Utils.var_from_returns(returns, alpha=0.05, dist="normal")
result = pd.DataFrame([result_dict])
print(result)