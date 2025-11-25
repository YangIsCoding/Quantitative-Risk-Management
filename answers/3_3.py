"""
題目 3.3: Higham 共變異數矩陣修正 (Higham Covariance Matrix Correction)

問題描述：
使用 Higham 演算法修正非正半定的共變異數矩陣。Higham 方法是另一種
尋找最接近原始矩陣的正半定矩陣的方法，通常比 Near-PSD 方法更精確，
但計算成本較高。該方法基於投影演算法。

目標：
1. 載入可能非正半定的共變異數矩陣
2. 應用 Higham 演算法進行修正
3. 確保結果為正半定矩陣
4. 輸出修正後的共變異數矩陣

解法流程：
1. 讀取 testout_1.3.csv 檔案（來自題目 1.3 的輸出）
2. 調用 Utils.higham_covariance() 進行 Higham 修正
3. 輸出修正後的正半定共變異數矩陣
"""

import pandas as pd
import library as Utils

# Test 3.3: Higham Covariance Matrix Correction
cin = pd.read_csv("../testfiles/data/testout_1.3.csv")

# Apply Higham method to correct covariance matrix
result = Utils.higham_covariance(cin)
print(result)