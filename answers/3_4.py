"""
題目 3.4: Higham 相關係數矩陣修正 (Higham Correlation Matrix Correction)

問題描述：
使用 Higham 演算法修正非正半定的相關係數矩陣。與 Higham 共變異數矩陣
修正類似，但需要額外確保修正後的矩陣保持相關係數矩陣的性質
（對角線元素為1，非對角元素在[-1,1]範圍內）。

目標：
1. 載入可能非正半定的相關係數矩陣
2. 應用 Higham 演算法進行修正
3. 確保結果為正半定且保持相關係數性質
4. 輸出修正後的相關係數矩陣

解法流程：
1. 讀取 testout_1.4.csv 檔案（來自題目 1.4 的輸出）
2. 調用 Utils.higham_correlation() 進行 Higham 修正
3. 輸出修正後的正半定相關係數矩陣
"""

import pandas as pd
import library as Utils

# Test 3.4: Higham Correlation Matrix Correction
cin = pd.read_csv("../testfiles/data/testout_1.4.csv")

# Apply Higham method to correct correlation matrix
result = Utils.higham_correlation(cin)
print(result)