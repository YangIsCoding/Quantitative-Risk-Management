"""
題目 2.1: 指數加權共變異數矩陣 (Exponentially Weighted Covariance Matrix)

問題描述：
計算指數加權共變異數矩陣，這種方法會給予較新的觀察值較高的權重，
較舊的觀察值較低的權重。λ (lambda) 參數控制衰減速度，λ=0.97 表示
相對較慢的衰減，給予歷史資料較多權重。

目標：
1. 載入時間序列資料
2. 使用指數加權方法計算共變異數矩陣
3. 設定 λ=0.97 作為衰減參數
4. 輸出標準化的指數加權共變異數矩陣

解法流程：
1. 讀取 test2.csv 資料
2. 調用 Utils.ew_cov_corr_normalized() 計算指數加權共變異數矩陣
3. 使用 λ=0.97 參數
4. 輸出共變異數矩陣結果
"""

import pandas as pd
import library as Utils

# Test 2.1: Exponentially Weighted Covariance Matrix
data = pd.read_csv("../testfiles/data/test2.csv")

# Calculate exponentially weighted covariance matrix with λ=0.97
cov_matrix, _ = Utils.ew_cov_corr_normalized(data, lam=0.97)

print(cov_matrix)