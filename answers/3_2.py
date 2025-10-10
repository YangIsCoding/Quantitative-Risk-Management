"""
作業目標 3.2: 非正定相關係數矩陣修正 - Near-PSD方法

背景:
承接第1.4題的Pairwise相關係數矩陣，該方法雖保留最多數據，
但結果矩陣可能不滿足正定性要求。

問題:
如何修正非正定的相關係數矩陣？

解法 - Rebonato-Jäckel (Near-PSD) 方法:
由於輸入已經是相關係數矩陣（對角線為1），
直接應用特徵值修正即可：

步驟:
1. 特徵值分解：R = VΛV'
2. 負特徵值歸零：Λ+ = max(Λ, 0)
3. 縮放步驟：計算縮放因子保持對角線為1
4. 重建矩陣：R+ = VΛ+V'
5. 標準化：確保 diag(R+) = 1

與3.1的差異:
- 3.1: 協方差→相關係數→修正→協方差（保持原始方差）
- 3.2: 相關係數→修正→相關係數（保持對角線為1）

應用:
- 配對交易中的相關性修正
- 因子模型的相關矩陣調整
- 統計套利策略的數值穩定性
"""

import pandas as pd
import numpy as np
from library import near_psd

if __name__ == "__main__":
    # 讀取第1.4題的Pairwise相關係數矩陣（可能非正定）
    cin = pd.read_csv("../testfiles/data/testout_1.4.csv")
    
    # 使用 Rebonato-Jäckel 方法修正為正定相關係數矩陣
    # 對於相關係數矩陣，保持對角線為1
    psd_matrix = near_psd(cin.values)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(psd_matrix, columns=cin.columns, index=cin.columns)
    
    print(result)