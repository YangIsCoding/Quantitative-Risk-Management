"""
作業目標 3.4: 非正定相關係數矩陣修正 - Higham方法

背景:
結合3.2和3.3的概念，使用Higham方法處理相關係數矩陣的正定性問題。

問題:
如何用Higham方法修正非正定的相關係數矩陣？

解法 - Higham方法處理相關係數矩陣:
對相關係數矩陣應用Higham交替投影算法

核心約束:
1. 正定性：eigenvalues ≥ 0
2. 相關矩陣性質：對角線元素 = 1
3. 範圍約束：非對角線元素 ∈ [-1, 1]

投影步驟:
1. PSD投影：R₊ = QΛ₊Q'（確保正定）
2. 單位對角線投影：diag(R) = 1（保持相關性質）
3. 迭代至收斂

與其他方法比較:
- 3.2 (Near-PSD): 單步特徵值修正
- 3.4 (Higham): 迭代交替投影
- 兩者都保持對角線為1

收斂性質:
- 單調收斂到最優解
- Frobenius範數下的最小距離
- 理論保證全域最優性

應用:
- 大規模相關矩陣修正
- 多因子模型的相關結構調整
- 統計套利策略的穩健性增強

實務考量:
相對於Near-PSD，Higham方法計算成本較高但精度更好。
"""

import pandas as pd
import numpy as np
from library import higham_nearestPSD

if __name__ == "__main__":
    # 讀取第1.4題的Pairwise相關係數矩陣（可能非正定）
    cin = pd.read_csv("../testfiles/data/testout_1.4.csv")
    
    # 使用 Higham 方法修正為正定相關係數矩陣
    # 透過交替投影保持對角線為1且確保正定性
    psd_matrix = higham_nearestPSD(cin.values)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(psd_matrix, columns=cin.columns, index=cin.columns)
    
    print(result)