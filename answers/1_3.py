"""
作業目標 1.3: 處理缺失值的協方差矩陣計算 - Pairwise Method

背景:
Skip Missing方法會丟失很多數據，當缺失值分散時尤其嚴重。

問題:
如何在不丟失太多數據的情況下處理缺失值？

解法 - Pairwise (可用案例分析):
每對變數單獨計算協方差，只使用該對變數都有值的觀察值。

優點: 保留更多信息，利用所有可用數據
缺點: 
1. 不同協方差項可能基於不同樣本大小
2. 結果矩陣可能不是正定的
3. 計算複雜度較高

數學公式:
Cov(Xi,Xj) 只使用 Xi 和 Xj 都不為缺失值的觀察值
每個協方差項可能基於不同的樣本大小 nij

注意: 此方法可能產生非正定矩陣，需要後續修正（見第3章）
"""

import pandas as pd
import numpy as np
from library import missing_cov

if __name__ == "__main__":
    # 讀取包含缺失值的測試數據
    data = pd.read_csv("../testfiles/data/test1.csv")
    
    # 使用 Pairwise 方法計算協方差矩陣
    # skipMiss=False: 每對變數單獨處理缺失值
    cov_matrix = missing_cov(data.values, skipMiss=False)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(cov_matrix, columns=data.columns, index=data.columns)

    print(result)