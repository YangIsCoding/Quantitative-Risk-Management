"""
題目 11.1: 資產歸因分析 (Asset Attribution Analysis)

問題描述：
進行投資組合的資產歸因分析，識別和量化各個資產對投資組合整體
績效的貢獻。資產歸因分析幫助投資經理了解哪些資產驅動了投資組合
的報酬，是績效評估的重要工具。

目標：
1. 載入投資組合報酬率資料和初始權重
2. 計算各資產對投資組合績效的貢獻
3. 分解總報酬為個別資產貢獻
4. 輸出詳細的歸因分析結果

解法流程：
1. 讀取 test11_1_returns.csv 檔案中的資產報酬率
2. 讀取 test11_1_weights.csv 檔案中的初始權重
3. 使用 Utils.asset_attribution() 進行資產歸因分析
4. 計算每個資產的績效貢獻
5. 輸出歸因分析結果
"""

import library as Utils
import pandas as pd

# Test 11.1: Asset Attribution Analysis
returns = pd.read_csv("../testfiles/data/test11_1_returns.csv")
init_weights = pd.read_csv("../testfiles/data/test11_1_weights.csv").values.flatten()

# Using the asset_attribution function from library
result = Utils.asset_attribution(returns, init_weights)
print(result)