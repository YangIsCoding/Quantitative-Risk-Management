"""
題目 11.2: 因子歸因分析 (Factor Attribution Analysis)

問題描述：
進行基於因子模型的歸因分析，將投資組合的績效歸因至特定風險因子。
因子歸因分析使用多因子模型來解釋投資組合報酬的來源，
幫助識別不同風險因子對績效的影響。

目標：
1. 載入資產報酬率、因子暴險度(beta)和因子報酬率
2. 建立多因子歸因模型
3. 計算各因子對投資組合績效的貢獻
4. 分解績效為因子貢獻和特殊報酬

解法流程：
1. 讀取 test11_2_stock_returns.csv 檔案中的資產報酬率
2. 讀取 test11_2_beta.csv 檔案中的因子暴險度
3. 讀取 test11_2_weights.csv 檔案中的投資組合權重
4. 讀取 test11_2_factor_returns.csv 檔案中的因子報酬率
5. 使用 Utils.factor_attribution() 進行因子歸因分析
6. 輸出因子歸因結果
"""

import library as Utils
import pandas as pd

# Test 11.2: Factor Attribution Analysis
asset_returns = pd.read_csv("../testfiles/data/test11_2_stock_returns.csv")
betas = pd.read_csv("../testfiles/data/test11_2_beta.csv", index_col=0)
init_weights = pd.read_csv("../testfiles/data/test11_2_weights.csv").values.flatten()
factor_returns = pd.read_csv("../testfiles/data/test11_2_factor_returns.csv")

# Using the factor_attribution function from library
result = Utils.factor_attribution(asset_returns, betas, init_weights, factor_returns)
print(result)