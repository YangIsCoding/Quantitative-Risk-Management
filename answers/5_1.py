"""
作業目標 5.1: 多變量常態分布模擬 - 正定協方差矩陣

背景:
在風險管理中，經常需要模擬相關的多變量隨機變數。
當協方差矩陣為正定時，可直接使用Cholesky分解進行模擬。

問題:
如何從給定的正定協方差矩陣生成相關的多變量常態隨機樣本？

解法 - Cholesky分解法:
1. 對協方差矩陣Σ進行Cholesky分解：Σ = LL'
2. 生成獨立標準常態隨機數 Z ~ N(0,I)
3. 轉換得到相關隨機數：X = ZL'

數學原理:
若 Z ~ N(0,I)，則 X = ZL' ~ N(0, LL') = N(0, Σ)

算法步驟:
1. 讀取正定協方差矩陣
2. Cholesky分解得到下三角矩陣L
3. 生成n×k獨立標準常態隨機數矩陣Z
4. 計算 X = ZL'
5. 驗證樣本協方差收斂到理論值

參數設定:
- 樣本數：100,000（確保統計收斂）
- 隨機種子：4（結果可重現）

應用場景:
- 投資組合回報模擬
- 市場風險情境分析
- 蒙地卡羅風險度量計算
- 壓力測試情境生成

統計驗證:
大數定律保證樣本協方差收斂到理論協方差矩陣。
"""

import pandas as pd
import numpy as np
from library import chol_psd_simple

if __name__ == "__main__":
    # 讀取正定協方差矩陣
    cin = pd.read_csv("../testfiles/data/test5_1.csv", header=None, skiprows=1).values.astype(float)
    
    # 設定隨機種子確保結果可重現
    np.random.seed(4)
    
    # 步驟1: Cholesky分解得到下三角矩陣L
    L = chol_psd_simple(cin)
    
    # 步驟2: 生成100,000個獨立標準常態隨機向量
    Z = np.random.randn(100000, 5)
    
    # 步驟3: 轉換為相關多變量常態分布 X = ZL'
    simulated_data = Z @ L.T
    
    # 步驟4: 計算樣本協方差矩陣（應接近理論值）
    result_matrix = np.cov(simulated_data.T)
    
    result = pd.DataFrame(result_matrix)
    print(result)