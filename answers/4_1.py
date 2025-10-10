"""
作業目標 4.1: 正定矩陣的Cholesky分解

背景:
正定矩陣可以分解為下三角矩陣L與其轉置的乘積：A = LL'
此分解在風險管理中極為重要，用於多變量隨機數生成。

問題:
如何對正定協方差矩陣進行Cholesky分解？

解法 - Cholesky分解:
對於正定矩陣A，存在唯一的下三角矩陣L使得A = LL'

數學公式:
L[i,j] = {
  √(A[i,i] - Σₖ₌₀ⁱ⁻¹ L[i,k]²)           if i = j
  (A[i,j] - Σₖ₌₀ʲ⁻¹ L[i,k]L[j,k]) / L[j,j]  if i > j  
  0                                      if i < j
}

算法步驟:
1. 檢查矩陣正定性（所有主對角子式>0）
2. 逐行計算下三角矩陣元素
3. 處理數值穩定性問題

應用場景:
- 蒙地卡羅模擬中的相關隨機數生成
- 多變量常態分布抽樣
- 投資組合風險分解
- 最優化問題的二次型處理

數值考量:
- 若矩陣接近奇異，可能出現數值不穩定
- 負特徵值會導致分解失敗
- 需要預先確保正定性（如使用第3章方法）

實務意義:
Cholesky分解是從獨立標準常態分布生成相關多變量分布的關鍵工具。
"""

import pandas as pd
import numpy as np
from library import chol_psd_simple

if __name__ == "__main__":
    # 讀取第3.1題的正定協方差矩陣
    cin = pd.read_csv("../testfiles/data/testout_3.1.csv").values
    
    # 使用 Cholesky 分解得到下三角矩陣 L
    # 滿足 A = LL'，用於生成相關多變量隨機數
    cout = chol_psd_simple(cin)
    
    # 將結果轉換為DataFrame輸出
    result = pd.DataFrame(cout)
    
    print(result)