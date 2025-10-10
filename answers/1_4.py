"""
作業目標 1.4: 處理缺失值的相關係數矩陣計算 - Pairwise Method

背景:
結合1.2和1.3的概念，使用Pairwise方法計算相關係數矩陣。

問題:
如何在保留最多數據的同時計算相關係數矩陣？

解法:
每對變數單獨計算相關係數，使用該對變數都有值的所有觀察值。

應用場景:
- 高頻金融數據中不同時間點的缺失
- 不同來源數據的合併
- 傳感器數據的間歇性故障

潛在問題:
1. 相關係數矩陣可能不是正定的
2. 不同相關係數基於不同樣本大小
3. 需要檢查是否滿足正定性要求（對於某些應用如投資組合優化）
"""

import pandas as pd
import numpy as np
from library import missing_cov

if __name__ == "__main__":
    # 讀取包含缺失值的測試數據
    data = pd.read_csv("../testfiles/data/test1.csv")
    
    # 使用 Pairwise 方法計算相關係數矩陣
    # skipMiss=False: 每對變數單獨處理缺失值
    # fun=np.corrcoef: 計算相關係數
    corr_matrix = missing_cov(data.values, skipMiss=False, fun=np.corrcoef)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(corr_matrix, columns=data.columns, index=data.columns)
    
    print(result)