"""
題目 3.2: 近正半定相關係數矩陣修正 (Near-PSD Correlation Matrix Correction)

問題描述：
修正非正半定的相關係數矩陣，確保其滿足正半定性質。相關係數矩陣
必須是正半定的，以確保在統計建模和風險分析中的數學一致性。
Near-PSD 方法尋找最接近原始矩陣的正半定矩陣。

目標：
1. 載入可能非正半定的相關係數矩陣
2. 應用 Near-PSD 修正方法
3. 確保修正後矩陣保持相關係數的性質（對角線為1）
4. 輸出修正後的相關係數矩陣

解法流程：
1. 讀取 testout_1.4.csv 檔案（來自題目 1.4 的輸出）
2. 調用 Utils.near_psd_correlation() 進行 Near-PSD 修正
3. 輸出修正後的正半定相關係數矩陣
"""

import pandas as pd
import library as Utils

# Test 3.2: Near-PSD Correlation Matrix Correction
cin = pd.read_csv("../testfiles/data/testout_1.4.csv")

# Apply Near-PSD correction to correlation matrix
result = Utils.near_psd_correlation(cin)
print(result)