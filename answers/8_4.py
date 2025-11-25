"""
題目 8.4: 使用常態分佈的預期損失計算 (Expected Shortfall using Normal Distribution)

問題描述：
計算預期損失（Expected Shortfall, ES），也稱為條件風險值（CVaR）。
ES 衡量當損失超過 VaR 時的平均損失，提供比 VaR 更全面的尾部風險
資訊，是風險管理中的重要指標。

目標：
1. 載入報酬率資料
2. 假設報酬率服從常態分佈
3. 計算 5% 信心水準的預期損失
4. 理解 ES 與 VaR 的關係和差異

解法流程：
1. 讀取 test7_1.csv 檔案中的報酬率資料
2. 提取 x1 欄位作為報酬率時間序列
3. 調用 Utils.es_normal() 計算常態分佈下的 ES
4. 設定 α=0.05（95% 信心水準）
5. 輸出 ES 計算結果
"""

import pandas as pd
import library as Utils

# Test 8.4: Expected Shortfall using Normal Distribution
data = pd.read_csv('../testfiles/data/test7_1.csv')
returns = data['x1']

# Calculate Expected Shortfall using normal distribution assumption
result_dict = Utils.es_normal(returns, alpha=0.05)
result = pd.DataFrame([result_dict])
print(result)