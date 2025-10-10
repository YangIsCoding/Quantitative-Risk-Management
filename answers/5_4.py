"""
作業目標 5.4: 多變量常態分布模擬 - Higham方法修正後模擬

背景:
比較5.3與5.4，探討不同矩陣修正方法對最終模擬結果的影響。
Higham方法提供理論最優的矩陣修正。

問題:
使用Higham方法修正非正定矩陣後，模擬結果與Near-PSD有何差異？

解法 - Higham方法修正流程:
1. 矩陣修正：使用Higham交替投影法
2. 正常模擬：對修正後矩陣進行Cholesky分解與模擬

Higham vs Near-PSD比較:
相同點：
- 都保持原始方差不變
- 都確保結果矩陣正定
- 都最小化某種距離度量

差異點：
- Higham：迭代收斂，理論最優解
- Near-PSD：單步計算，快速近似解
- Higham：計算成本較高
- Near-PSD：實務中更常用

算法特徵:
Higham方法：
- 交替投影到PSD錐與對角線約束集
- 單調收斂到全域最優
- Frobenius範數下的最小距離

收斂保證:
- 理論證明的全域收斂性
- 投影操作的非擴張性
- 不動點定理保證

實務應用:
- 對精度要求極高的場合
- 學術研究中的基準比較
- 大型機構的風險建模
- 監管報告的嚴格要求

數值比較:
通過與5.3的結果比較，可以量化兩種方法的差異。
"""

import pandas as pd
import numpy as np
from library import higham_nearestPSD, chol_psd_simple

if __name__ == "__main__":
    # 讀取非正定協方差矩陣（與5.3相同輸入）
    cin = pd.read_csv("../testfiles/data/test5_3.csv", header=None, skiprows=1).values.astype(float)
    
    # 步驟1: 使用 Higham 方法修正為正定矩陣
    # 交替投影法獲得理論最優解
    fixed_matrix = higham_nearestPSD(cin)
    
    # 設定隨機種子確保與5.3可比較
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