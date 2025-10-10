"""
作業目標 1.2: 處理缺失值的相關係數矩陣計算 - Skip Missing Method

背景:
相關係數矩陣是協方差矩陣的標準化版本，值域在[-1,1]之間，更容易解釋變數間的線性關係強度。

問題:
當數據中有缺失值時，如何計算相關係數矩陣？

解法:
與1.1相同，使用Skip Missing方法，但計算相關係數而非協方差。

數學公式:
Corr(X,Y) = Cov(X,Y) / (σx * σy)
其中 σx = sqrt(Var(X)), σy = sqrt(Var(Y))

相關係數的解釋:
+1: 完全正相關
 0: 無線性相關
-1: 完全負相關
"""

import pandas as pd
import numpy as np
from library import missing_cov

if __name__ == "__main__":
    # 讀取包含缺失值的測試數據
    data = pd.read_csv("../testfiles/data/test1.csv")
    
    # 使用 Skip Missing 方法計算相關係數矩陣
    # fun=np.corrcoef: 計算相關係數而非協方差
    corr_matrix = missing_cov(data.values, skipMiss=True, fun=np.corrcoef)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(corr_matrix, columns=data.columns, index=data.columns)
    
    print(result)