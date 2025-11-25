"""
題目 8.5: 使用 t 分佈的預期損失計算 (Expected Shortfall using t-Distribution)

問題描述：
使用 t 分佈假設計算預期損失（ES）。相較於常態分佈，t 分佈的厚尾特性
使得 ES 的估計更能反映極端市場條件下的真實風險，特別適合金融市場
的風險評估。

目標：
1. 載入報酬率資料
2. 假設報酬率服從 t 分佈
3. 計算 5% 信心水準的預期損失
4. 比較與常態分佈 ES 的差異

解法流程：
1. 讀取 test7_2.csv 檔案中的報酬率資料
2. 提取 x1 欄位作為報酬率時間序列
3. 調用 Utils.es_t() 計算 t 分佈下的 ES
4. 設定 α=0.05（95% 信心水準）
5. 輸出 ES 計算結果
"""

import pandas as pd
import library as Utils

# Test 8.5: Expected Shortfall using t-Distribution
data = pd.read_csv("../testfiles/data/test7_2.csv")
returns = data["x1"]

# Calculate Expected Shortfall using t-distribution assumption
result_dict = Utils.es_t(returns, alpha=0.05)
result = pd.DataFrame([result_dict])
print(result)