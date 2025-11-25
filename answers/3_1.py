"""
題目 3.1: 近正半定共變異數矩陣修正 (Near-PSD Covariance Matrix Correction - Rebonato-Jäckel 方法)

問題描述：
修正非正半定的共變異數矩陣，使其成為最接近原始矩陣的正半定矩陣。
正半定性是共變異數矩陣必須滿足的數學性質，確保所有特徵值非負，
避免在風險建模中出現負變異數等不合理結果。

目標：
1. 載入可能非正半定的共變異數矩陣
2. 應用 Near-PSD 修正方法
3. 使用 Rebonato-Jäckel 演算法確保正半定性
4. 輸出修正後的共變異數矩陣

解法流程：
1. 讀取 testout_1.3.csv 檔案（來自題目 1.3 的輸出）
2. 調用 Utils.near_psd_covariance() 進行 Near-PSD 修正
3. 輸出修正後的正半定共變異數矩陣
"""

import pandas as pd
import library as Utils

# Test 3.1: Near-PSD Covariance Matrix Correction (Rebonato-Jäckel method)
cin = pd.read_csv("../testfiles/data/testout_1.3.csv")

# Apply Near-PSD correction to covariance matrix
result = Utils.near_psd_covariance(cin)
print(result)