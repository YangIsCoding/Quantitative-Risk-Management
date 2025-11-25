"""
題目 7.2: t 分佈參數估計 (t-Distribution Parameter Estimation)

問題描述：
使用最大概似估計法來估計 t 分佈的參數（位置參數 μ、尺度參數 σ、
自由度 ν）。t 分佈比常態分佈有更厚的尾部，更適合描述金融市場中
極端事件發生的機率。

目標：
1. 載入樣本資料
2. 使用 MLE 方法估計 t 分佈參數
3. 估計位置參數 μ（類似均值）
4. 估計尺度參數 σ（類似標準差）
5. 估計自由度 ν（控制尾部厚度）

解法流程：
1. 讀取 test7_2.csv 檔案中的樣本資料
2. 提取 x1 欄位的數值
3. 使用 scipy.stats.t.fit() 進行參數估計
4. 輸出估計的 μ、σ 和 ν 參數值
"""

import pandas as pd
import numpy as np
from scipy import stats

# Test 7.2: t-Distribution Parameter Estimation
data = pd.read_csv("../testfiles/data/test7_2.csv")["x1"].values

# Use scipy.stats.t to fit t-distribution
nu, mu, sigma = stats.t.fit(data)

# Prepare result
result = pd.DataFrame({
    "mu": [mu],
    "sigma": [sigma], 
    "nu": [nu]
})

print(result)