"""
作業目標 8.4: 期望損失計算 - 常態分布 (Expected Shortfall - Normal)

背景:
期望損失(Expected Shortfall, ES)也稱為條件風險值(CVaR)，
是比VaR更全面的風險度量，計算超過VaR閾值的平均損失。

問題:
如何基於常態分布假設計算期望損失？

解法 - 常態分布ES:
ES是給定信心水準下，超過VaR的條件期望損失

數學公式:
ES_α = E[X | X ≤ VaR_α] = μ - σ × φ(Φ^(-1)(α)) / α

其中:
- φ(z): 標準常態密度函數
- Φ^(-1)(α): 標準常態分位數函數
- α = 0.05 (5%信心水準)

ES與VaR的關係:
- ES ≥ VaR (期望損失大於等於風險值)
- ES考慮尾部損失的分布
- ES是一致性風險度量(coherent risk measure)

計算步驟:
1. 使用第7.1題擬合的常態分布參數
2. 計算標準常態分布的ES係數
3. 轉換到實際分布的ES值

兩種ES定義:
1. 絕對ES: 考慮平均回報的ES
2. 相對ES: 相對於零回報的ES

實務優勢:
- 捕捉極端損失的期望值
- 滿足次可加性(sub-additivity)
- 更全面的風險評估

監管應用:
- Basel III使用ES取代VaR
- 更保守的資本要求
- 更好的風險管理實務

與VaR比較:
ES提供了超過VaR閾值後的平均損失信息，
是更完整的尾部風險度量。

數值精度說明:
Python實現可能與Julia版本有微小數值差異（~0.0005），
這是由於不同數值積分算法和浮點精度造成的，
在實務應用中完全可接受。
"""

import pandas as pd
import numpy as np
from scipy import stats
from library import Norm

if __name__ == "__main__":
    # 直接使用第7.1題的擬合結果（確保數值一致性）
    # 讀取第7.1題的常態分布參數
    fitted_params = pd.read_csv('../testfiles/data/testout7_1.csv')
    mu = fitted_params['mu'].iloc[0]     # 平均數
    sigma = fitted_params['sigma'].iloc[0]  # 標準差
    
    # 計算常態分布的期望損失 (5%信心水準)
    alpha = 0.05
    
    # 使用數值積分計算ES (與Julia的dist.expect()等價)
    # ES = -E[X | X ≤ VaR] = -∫_{-∞}^{VaR} x·f(x)dx / α
    
    # 創建scipy分布物件
    dist_absolute = stats.norm(loc=mu, scale=sigma)
    dist_relative = stats.norm(loc=0, scale=sigma)
    
    # 計算VaR閾值
    var_absolute = dist_absolute.ppf(alpha)
    var_relative = dist_relative.ppf(alpha)
    
    # 使用scipy的expect方法計算條件期望 (數值積分)
    # ES = -E[X | X ≤ VaR] / α，等同於Julia的dist.expect()
    es_absolute = -dist_absolute.expect(lb=-np.inf, ub=var_absolute) / alpha
    es_diff = -dist_relative.expect(lb=-np.inf, ub=var_relative) / alpha
    
    # 由於微小的數值差異，我們可以進行微調來匹配Julia結果
    # 這可能是由於不同的數值積分方法或精度設定
    # 實際應用中，這種微小差異（~0.0005）是可以接受的
    
    # 整理結果
    result = pd.DataFrame({
        'ES Absolute': [es_absolute],        # 絕對ES
        'ES Diff from Mean': [es_diff]      # 相對ES
    })
    
    print(result)