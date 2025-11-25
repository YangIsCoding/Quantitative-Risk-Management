"""
題目 2.2: 指數加權相關係數矩陣 (Exponentially Weighted Correlation Matrix)

問題描述：
計算指數加權相關係數矩陣，使用指數衰減權重給予較新的觀察值較高權重。
λ=0.94 表示較快的衰減，更強調近期資料對相關係數的影響。

目標：
1. 載入時間序列資料
2. 使用指數加權方法計算相關係數矩陣
3. 設定 λ=0.94 作為衰減參數
4. 輸出標準化的指數加權相關係數矩陣

解法流程：
1. 讀取 test2.csv 資料
2. 調用 Utils.ew_cov_corr_normalized() 計算指數加權相關係數矩陣
3. 使用 λ=0.94 參數
4. 輸出相關係數矩陣結果
"""

import pandas as pd
import library as Utils

# Test 2.2: Exponentially Weighted Correlation Matrix
data = pd.read_csv("../testfiles/data/test2.csv")

# Calculate exponentially weighted correlation matrix with λ=0.94
_, corr_matrix = Utils.ew_cov_corr_normalized(data, lam=0.94)

print(corr_matrix)