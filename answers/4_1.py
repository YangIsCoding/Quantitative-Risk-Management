"""
題目 4.1: 正半定矩陣的 Cholesky 分解 (Cholesky Decomposition of PSD Matrix)

問題描述：
對正半定矩陣進行 Cholesky 分解，將矩陣 A 分解為 A = L @ L.T 的形式，
其中 L 是下三角矩陣。Cholesky 分解在蒙地卡羅模擬、風險建模等領域
非常重要，用於生成相關隨機變數。

目標：
1. 載入正半定矩陣
2. 進行 Cholesky 分解得到下三角矩陣 L
3. 處理可能的數值誤差（非嚴格正定的情況）
4. 輸出下三角矩陣 L

解法流程：
1. 讀取 testout_3.1.csv 檔案（來自題目 3.1 的修正後矩陣）
2. 嘗試標準 Cholesky 分解
3. 如果失敗，使用特徵值分解的替代方法
4. 確保特徵值為正數（處理數值誤差）
5. 輸出下三角矩陣 L
"""

import pandas as pd
import numpy as np

# Test 4.1: Cholesky Decomposition of PSD Matrix
cin = pd.read_csv("../testfiles/data/testout_3.1.csv").values

# Perform Cholesky decomposition to get lower triangular matrix L
# such that A = L @ L.T
try:
    L = np.linalg.cholesky(cin)
except np.linalg.LinAlgError:
    # If not positive definite, use eigenvalue method
    w, V = np.linalg.eigh(cin)
    w = np.maximum(w, 1e-8)  # Ensure positive eigenvalues
    L = V @ np.diag(np.sqrt(w))

result = pd.DataFrame(L)
print(result)