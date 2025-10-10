"""
作業目標 5.2: 多變量常態分布模擬 - 半正定協方差矩陣

背景:
實務中的協方差矩陣可能是半正定的（特徵值≥0但部分=0），
此時標準Cholesky分解會失敗，需要修正的分解方法。

問題:
如何處理半正定協方差矩陣的多變量常態分布模擬？

解法 - 修正Cholesky分解法:
使用能處理半正定矩陣的Cholesky分解變種：
1. 檢測並處理零或負特徵值
2. 對修正後的矩陣進行分解
3. 生成相應的隨機樣本

算法特點:
- 自動處理數值精度問題
- 將微小負特徵值設為0
- 確保分解的數值穩定性

與5.1的差異:
- 5.1: 嚴格正定矩陣（所有特徵值>0）
- 5.2: 半正定矩陣（特徵值≥0）

技術實現:
chol_psd_simple函數內建半正定矩陣處理邏輯：
- 特徵值分解 + 負值歸零
- 修正的Cholesky算法
- 數值穩定性保證

實務意義:
金融數據中常出現半正定協方差矩陣：
- 高度相關的資產
- 維度詛咒導致的估計誤差
- 缺失數據的影響

應用場景:
- 降維後的風險因子模擬
- 受約束的投資組合分析
- 主成分風險模型
"""

import pandas as pd
import numpy as np
from library import chol_psd_simple

if __name__ == "__main__":
    # 讀取半正定協方差矩陣
    cin = pd.read_csv("../testfiles/data/test5_2.csv", header=None, skiprows=1).values.astype(float)
    
    # 設定隨機種子確保結果可重現
    np.random.seed(4)
    
    # 使用修正的Cholesky分解處理半正定矩陣
    # chol_psd_simple自動處理零特徵值和數值穩定性問題
    L = chol_psd_simple(cin)
    
    # 生成獨立標準常態隨機數
    Z = np.random.randn(100000, 5)
    
    # 轉換為相關多變量分布
    simulated_data = Z @ L.T
    
    # 計算樣本協方差矩陣
    result_matrix = np.cov(simulated_data.T)
    
    result = pd.DataFrame(result_matrix)
    print(result)