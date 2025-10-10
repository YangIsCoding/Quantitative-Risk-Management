"""
作業目標 2.2: 指數加權相關係數矩陣計算 (EWMA Correlation)

背景:
相關係數矩陣是標準化的協方差矩陣，移除了量綱影響，更適合比較不同資產間的關聯性。

問題:
如何獲得指數加權的相關係數矩陣？

解法:
1. 先計算指數加權協方差矩陣
2. 提取對角線元素（方差）計算標準差
3. 標準化得到相關係數矩陣

數學公式:
Corr(i,j) = Cov(i,j) / (σi * σj)
其中 σi = √Var(i) = √Cov(i,i)

參數選擇:
λ=0.94: 比協方差更敏感的衰減因子
- 相關係數變化通常比方差變化更快
- 較小的λ能更好捕捉關聯性的結構變化

應用場景:
- 配對交易策略
- 資產配置中的相關性分析
- 系統性風險評估
- 投資組合分散化效果評估

注意事項:
相關係數矩陣的對角線應該為1，非對角線元素在[-1,1]範圍內。
"""

import pandas as pd
import numpy as np
from library import ewCovar

if __name__ == "__main__":
    # 讀取測試數據
    data = pd.read_csv("../testfiles/data/test2.csv")
    
    # 計算指數加權協方差矩陣，使用較小的λ=0.94
    # 較小的λ使相關係數對近期變化更敏感
    cout = ewCovar(data.values, 0.94)
    
    # 轉換為相關係數矩陣
    # 1. 計算標準差的倒數：1/σi
    sd = 1.0 / np.sqrt(np.diag(cout))
    # 2. 標準化：Corr = D^(-1) * Cov * D^(-1)，其中D是標準差對角矩陣
    corr_matrix = np.diag(sd) @ cout @ np.diag(sd)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(corr_matrix, columns=data.columns, index=data.columns)
    
    print(result)