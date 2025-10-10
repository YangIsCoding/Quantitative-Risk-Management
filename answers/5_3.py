"""
作業目標 5.3: 多變量常態分布模擬 - 非正定矩陣修正後模擬

背景:
實際數據中的協方差矩陣估計可能不滿足正定性，
必須先修正為正定矩陣後才能進行模擬。

問題:
如何處理非正定協方差矩陣的多變量常態分布模擬？

解法 - 兩步驟程序:
1. 矩陣修正：使用Near-PSD方法修正為正定矩陣
2. 正常模擬：對修正後矩陣進行Cholesky分解與模擬

技術流程:
第一步：非正定 → 正定
- 輸入：具有負特徵值的矩陣
- 使用：Rebonato-Jäckel Near-PSD算法
- 輸出：最接近的正定矩陣

第二步：正定 → 模擬
- 輸入：修正後的正定矩陣
- 使用：標準Cholesky分解
- 輸出：相關多變量常態隨機樣本

實務考量:
- 修正過程保持原始方差不變
- 最小化矩陣距離（Frobenius範數）
- 確保數值穩定性

應用場景:
- 歷史數據估計的協方差矩陣
- 小樣本導致的非正定問題
- 合併不同來源數據的協方差
- 極端市場條件下的風險建模

品質控制:
- 驗證修正後矩陣的正定性
- 檢查樣本協方差收斂性
- 比較修正前後的差異

統計意義:
這種方法確保了模擬的有效性，同時最大程度保持原始風險結構。
"""

import pandas as pd
import numpy as np
from library import near_psd, chol_psd_simple

if __name__ == "__main__":
    # 讀取非正定協方差矩陣
    cin = pd.read_csv("../testfiles/data/test5_3.csv", header=None, skiprows=1).values.astype(float)
    
    # 步驟1: 使用 Near-PSD 方法修正為正定矩陣
    # 保持原始方差，修正負特徵值
    fixed_matrix = near_psd(cin)
    
    # 設定隨機種子確保結果可重現
    np.random.seed(4)
    
    # 步驟2: 對修正後的正定矩陣進行Cholesky分解
    L = chol_psd_simple(fixed_matrix)
    
    # 步驟3: 生成獨立標準常態隨機數
    Z = np.random.randn(100000, 5)
    
    # 步驟4: 轉換為相關多變量常態分布
    simulated_data = Z @ L.T
    
    # 步驟5: 計算樣本協方差矩陣
    result_matrix = np.cov(simulated_data.T)
    
    result = pd.DataFrame(result_matrix)
    print(result)