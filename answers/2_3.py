"""
作業目標 2.3: 混合指數加權協方差矩陣 (混合λ參數)

背景:
在實務中，方差和相關係數可能有不同的持續性特徵：
- 方差（波動率）變化較快，需要較小的λ
- 相關係數結構變化較慢，可以用較大的λ

問題:
如何結合不同衰減參數的優勢？

解法 - 混合EWMA方法:
1. 用λ=0.97計算相關係數矩陣（長記憶）
2. 用λ=0.94計算方差（短記憶）
3. 重新組合得到最終協方差矩陣

數學公式:
Σ = D_var * R_corr * D_var
其中：
- D_var: 來自λ=0.94的標準差對角矩陣
- R_corr: 來自λ=0.97的相關係數矩陣

實現步驟:
1. 計算EW相關係數（λ=0.97）-> 提取標準差sd1
2. 計算EW協方差（λ=0.94）-> 提取標準差的倒數sd
3. 組合：D(sd1) * D(sd) * Cov * D(sd) * D(sd1)

優點:
1. 方差快速適應波動率叢聚
2. 相關係數保持結構穩定性
3. 避免過度反應帶來的雜訊

應用:
- 複雜投資組合的風險建模
- 多因子風險模型
- 結構化產品定價

實務意義:
這種方法在2008年金融危機等極端市場條件下表現更穩健。
"""

import pandas as pd
import numpy as np
from library import ewCovar

if __name__ == "__main__":
    # 讀取測試數據
    data = pd.read_csv("../testfiles/data/test2.csv")
    
    # 步驟1: 用λ=0.97計算相關係數矩陣（長記憶）
    cout = ewCovar(data.values, 0.97)
    sd1 = np.sqrt(np.diag(cout))  # 提取標準差
    
    # 步驟2: 用λ=0.94計算方差（短記憶，對波動更敏感）
    cout = ewCovar(data.values, 0.94)
    sd = 1.0 / np.sqrt(np.diag(cout))  # 計算標準差的倒數
    
    # 步驟3: 重新組合 - 結合短記憶方差和長記憶相關係數
    # 公式: Σ = D(sd1) * D(sd^-1) * Cov * D(sd^-1) * D(sd1)
    # 這等效於將λ=0.94的協方差矩陣重新縮放到λ=0.97的方差水平
    result_matrix = np.diag(sd1) @ np.diag(sd) @ cout @ np.diag(sd) @ np.diag(sd1)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(result_matrix, columns=data.columns, index=data.columns)
   
    print(result)