"""
題目 10.1: 基本風險平價投資組合（等風險預算）(Basic Risk Parity Portfolio - Equal Risk Budget)

問題描述：
建立風險平價投資組合，確保每個資產對投資組合總風險的貢獻相等。
風險平價是一種重要的投資組合建構方法，關注風險分散而非資本分散，
能夠更好地平衡各資產的風險貢獻。

目標：
1. 載入資產共變異數矩陣
2. 計算風險平價權重
3. 確保每個資產的風險貢獻相等
4. 輸出最優投資組合權重

解法流程：
1. 讀取 test5_2.csv 檔案中的共變異數矩陣
2. 使用 Utils.risk_parity_csd() 計算風險平價權重
3. 解決約束優化問題以達成等風險貢獻
4. 輸出投資組合權重向量
"""

import library as Utils
import pandas as pd

# Test10.1: Basic Risk Parity Portfolio (Equal Risk Budget)
cov = pd.read_csv("../testfiles/data/test5_2.csv").values

# Using the risk_parity_csd function from library
weights = Utils.risk_parity_csd(cov)
print(weights)