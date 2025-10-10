"""
作業目標 7.1: 常態分布參數估計 (Normal Distribution Fitting)

背景:
常態分布是金融建模的基礎分布，許多風險模型都假設回報率遵循常態分布。
參數估計是建立機率模型的第一步。

問題:
如何從樣本數據估計常態分布的參數？

解法 - 最大概似估計法 (MLE):
對於常態分布 N(μ, σ²)，MLE估計量為：

參數估計:
- μ̂ = (1/n) Σᵢ₌₁ⁿ xᵢ  (樣本平均數)
- σ̂² = (1/n) Σᵢ₌₁ⁿ (xᵢ - μ̂)²  (樣本方差)

估計特性:
- μ̂ 是無偏估計量：E[μ̂] = μ
- σ̂² 是有偏估計量，但一致性估計量
- 大樣本下具有漸近常態性

應用場景:
- 風險度量的基礎分布
- 蒙地卡羅模擬的參數輸入
- 投資組合理論的收益分布
- 期權定價模型的假設檢驗

統計檢驗:
估計後通常需要進行：
- 常態性檢驗（Jarque-Bera、Shapiro-Wilk）
- 參數穩定性檢驗
- 殘差分析

實務考量:
- 金融數據常呈現厚尾現象
- 波動率叢聚問題
- 結構性突破的影響

模型限制:
- 假設數據獨立同分布
- 忽略時間序列的條件異方差
- 可能低估極端事件機率

後續應用:
擬合參數將用於風險度量計算，如VaR和Expected Shortfall。
"""

import pandas as pd
import numpy as np
from library import Norm

if __name__ == "__main__":
    # 讀取數據進行常態分布擬合
    data = pd.read_csv('../testfiles/data/test7_1.csv')
    x_values = data['x1'].values
    
    # 使用library中的Norm類別進行最大概似估計
    normal_model = Norm()
    normal_model.fit(x_values)
    
    # 提取擬合的參數
    mu = normal_model.fitted_parameters[0]     # 平均數
    sigma = normal_model.fitted_parameters[1]  # 標準差
    
    # 整理結果
    result = pd.DataFrame({
        'mu': [mu],
        'sigma': [sigma]
    })
    
    print(result)