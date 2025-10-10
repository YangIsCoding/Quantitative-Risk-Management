"""
作業目標 1.1: 處理缺失值的協方差矩陣計算 - Skip Missing Method

背景:
在金融數據分析中，經常遇到缺失值問題。計算協方差矩陣時需要妥善處理這些缺失值。

問題:
當數據中有缺失值時，如何計算可靠的協方差矩陣？

解法 1 - Skip Missing (listwise deletion):
移除任何包含缺失值的整個觀察值（行），只使用完整的觀察值計算協方差。

優點: 計算簡單，結果一致
缺點: 可能丟失大量數據，特別是當缺失值分散時

數學公式:
Cov(X,Y) = E[(X-μx)(Y-μy)] = (1/n)Σ(xi-μx)(yi-μy)
其中只使用xi和yi都不為缺失值的觀察值
"""

import pandas as pd
import numpy as np
from library import missing_cov

if __name__ == "__main__":
    # 讀取包含缺失值的測試數據
    data = pd.read_csv("../testfiles/data/test1.csv")
    
    # 使用 Skip Missing 方法計算協方差矩陣
    # skipMiss=True: 移除包含任何缺失值的整行
    cov_matrix = missing_cov(data.values, skipMiss=True)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(cov_matrix, columns=data.columns, index=data.columns)
    
    print(result)