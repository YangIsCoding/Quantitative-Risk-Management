"""
作業目標 6.2: 對數回報率計算 (Log Returns)

背景:
對數回報率（連續複利回報率）在金融理論中具有優良的統計性質，
特別適合風險建模和時間序列分析。

問題:
如何從價格時間序列計算對數回報率？

解法 - 對數回報率公式:
rt = ln(Pt/Pt-1) = ln(Pt) - ln(Pt-1)

數學特性:
- 時間可加性：多期回報 = 單期回報之和
- 對稱性：上漲50%和下跌33.3%的對數回報大小相等
- 常態性：更接近常態分布（統計分析優勢）
- 連續性：便於微積分運算

與離散回報率的關係:
- 小幅變化時：rt^log ≈ rt^discrete
- 大幅變化時：存在顯著差異
- 轉換公式：rt^discrete = exp(rt^log) - 1

應用優勢:
- 風險模型（VaR、ES）
- 時間序列分析（ARIMA、GARCH）
- 期權定價模型（Black-Scholes）
- 長期投資分析

統計性質:
- 更接近常態分布
- 方差穩定性較好
- 便於參數估計
- 支持複利計算

實務應用:
- 波動率建模
- 風險度量計算
- 量化策略回測
- 學術研究標準

計算考量:
- 處理零價格和負價格
- 確保價格數據品質
- 處理停牌和缺失數據

理論基礎:
幾何布朗運動假設下，對數價格遵循常態分布，
使得對數回報率成為金融建模的自然選擇。
"""

import pandas as pd
import numpy as np
from library import return_calculate

if __name__ == "__main__":
    # 讀取價格數據
    prices = pd.read_csv("../testfiles/data/test6.csv")
    
    # 計算對數回報率（連續複利回報率）
    # rt = ln(Pt/Pt-1) = ln(Pt) - ln(Pt-1)
    returns = return_calculate(prices, method="LOG", dateColumn="Date")
    
    print(returns.head())