"""
作業目標 3.1: 非正定矩陣修正 - Rebonato-Jäckel (Near-PSD) 方法

背景:
在第1章的Pairwise方法中，由於不同協方差項基於不同樣本，結果矩陣可能不是
正定的（PSD）。非正定矩陣會導致數值問題，如Cholesky分解失敗、
投資組合優化無解等。

問題:
如何將非正定矩陣調整為最接近的正定矩陣，同時保持原始方差？

解法 - Rebonato-Jäckel (Near-PSD) 方法:
1. 提取標準差（保存原始方差）
2. 轉換為相關係數矩陣（對角線為1）
3. 特徵值分解，將負特徵值設為0
4. 縮放步驟：計算縮放因子以保持對角線為1
5. 重建相關係數矩陣並標準化
6. 轉回協方差空間（恢復原始方差）

數學公式:
設 A 為輸入矩陣，σ = √diag(A)
1. R = A / (σσ')
2. R = VΛV', Λ+ = max(Λ, 0)
3. t_i = 1 / Σj (Vij)^2 λj+
4. B = √T V √Λ+
5. C = BB', 標準化使 diag(C) = 1
6. Σ = diag(σ) C diag(σ)

重要特性:
- 保持原始矩陣的方差（對角線元素）
- 最小化Frobenius範數下的距離
- 結果矩陣確保為正定的

應用:
- 風險管理中的協方差矩陣修正
- 投資組合優化的前處理
- 蒙地卡羅模擬的数值穩定性
"""

import pandas as pd
import numpy as np
from library import near_psd

if __name__ == "__main__":
    # 讀取第1.3題的輸出（Pairwise協方差矩陣，可能非正定）
    cin = pd.read_csv("../testfiles/data/testout_1.3.csv")
    
    # 使用 Rebonato-Jäckel (Near-PSD) 方法修正為正定矩陣
    # 此方法保持原始方差（對角線元素）不變
    psd_matrix = near_psd(cin.values)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(psd_matrix, columns=cin.columns, index=cin.columns)
    
    print(result)