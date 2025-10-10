"""
作業目標 3.3: 非正定矩陣修正 - Higham方法

背景:
除了Rebonato-Jäckel方法外，Higham方法是另一種重要的
正定矩陣修正技術，特別適用於需要保持原始方差的情況。

問題:
如何使用Higham方法修正非正定協方差矩陣？

解法 - Higham最近正定矩陣方法:
透過交替投影技術尋找最接近的正定矩陣

演算法步驟:
1. 初始化：X₀ = A（輸入矩陣）
2. 交替投影：
   - Yₖ = X₀ + δₖ（投影到PSD空間）
   - Xₖ₊₁ = P_S(Yₖ)（投影到原始對角線空間）
3. 迭代直到收斂

核心特性:
- 保持原始方差（對角線元素）
- 最小化Frobenius範數距離
- 保證收斂到全域最優解

數學原理:
投影操作：P_PSD(A) = QΛ₊Q'，其中Λ₊ = max(Λ, τ)
對角線投影：diag(X) = diag(A)

與Rebonato-Jäckel的比較:
- Higham: 迭代收斂，理論最優
- RJ: 單步計算，實務快速
- 兩者都保持原始方差

應用:
- 大型投資組合的協方差矩陣修正
- 結構化產品的風險建模
- 高維度風險因子分析
"""

import pandas as pd
import numpy as np
from library import higham_nearestPSD

if __name__ == "__main__":
    # 讀取第1.3題的Pairwise協方差矩陣（可能非正定）
    cin = pd.read_csv("../testfiles/data/testout_1.3.csv")
    
    # 使用 Higham 方法修正為正定矩陣
    # 此方法透過交替投影保持原始方差不變
    psd_matrix = higham_nearestPSD(cin.values)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(psd_matrix, columns=cin.columns, index=cin.columns)
    
    print(result)