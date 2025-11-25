"""
題目 7.1: 常態分佈參數估計 (Normal Distribution Parameter Estimation)

問題描述：
使用最大概似估計法（Maximum Likelihood Estimation, MLE）來估計
常態分佈的參數（均值 μ 和標準差 σ）。這是統計學和風險管理中
基礎且重要的參數估計技術。

目標：
1. 載入樣本資料
2. 使用 MLE 方法估計常態分佈參數
3. 計算樣本均值（μ 的估計值）
4. 計算樣本標準差（σ 的估計值）

解法流程：
1. 讀取 test7_1.csv 檔案中的樣本資料
2. 提取 x1 欄位的數值
3. 調用 Utils.fit_normal() 進行參數估計
4. 輸出估計的 μ 和 σ 參數值
"""

import pandas as pd
import library as Utils

# Test 7.1: Normal Distribution Parameter Estimation
data = pd.read_csv('../testfiles/data/test7_1.csv')
x_values = data['x1'].values

# Use the fit_normal function from library
mu, sigma = Utils.fit_normal(x_values)

# Prepare result
result = pd.DataFrame({
    'mu': [mu],
    'sigma': [sigma]
})

print(result)