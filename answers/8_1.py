"""
作業目標 8.1: 常態分布風險值計算 (Normal VaR)

背景:
風險值(VaR)是衡量金融風險的核心指標，表示在給定信心水準下，
投資組合在特定時間內可能遭受的最大損失。

問題:
如何基於常態分布假設計算VaR？

解法 - 參數法VaR:
假設回報率遵循常態分布，使用擬合參數計算分位數

VaR計算公式:
VaR_α = -F^(-1)(α) = -Φ^(-1)(α; μ, σ)
其中 α = 0.05 (5%機率)

兩種VaR定義:
1. 絕對VaR: 考慮平均回報
   VaR_absolute = -(μ + σΦ^(-1)(0.05))
   
2. 相對VaR: 相對於零回報的VaR
   VaR_relative = -σΦ^(-1)(0.05)

參數估計:
使用第7.1題擬合的常態分布參數：
- μ: 樣本平均數（期望回報）
- σ: 樣本標準差（波動率）

統計意義:
- VaR表示5%機率下的最大損失
- 95%信心水準的風險度量
- 左尾風險的量化指標

應用差異:
- 絕對VaR: 考慮平均回報的影響
- 相對VaR: 純粹的波動率風險

實務考量:
- 常態假設可能低估極端風險
- 需要定期更新參數估計
- 回測驗證VaR模型準確性

監管應用:
- 資本適足性要求
- 風險限額設定
- 績效評估基準

模型限制:
- 忽略厚尾和偏態特性
- 假設參數時間穩定
- 可能低估系統性風險

後續改進:
第8.2和8.3將使用t分布和歷史模擬法改善VaR估計。
"""

import pandas as pd
import numpy as np
from scipy import stats
from library import Norm

if __name__ == "__main__":
    # 讀取數據，使用第7.1題相同的數據
    data = pd.read_csv('../testfiles/data/test7_1.csv')['x1'].values
    
    # 使用library擬合常態分布
    normal_model = Norm()
    normal_model.fit(data)
    
    # 提取擬合參數
    mu = normal_model.fitted_parameters[0]     # 平均數
    sigma = normal_model.fitted_parameters[1]  # 標準差
    
    # 計算VaR (5%信心水準)
    # 使用負號因為VaR表示損失（正值）
    var_absolute = -stats.norm.ppf(0.05, loc=mu, scale=sigma)
    var_diff = -stats.norm.ppf(0.05, loc=0, scale=sigma)
    
    # 整理結果
    result = pd.DataFrame({
        'VaR Absolute': [var_absolute],        # 絕對VaR
        'VaR Diff from Mean': [var_diff]      # 相對VaR
    })
    
    print(result)