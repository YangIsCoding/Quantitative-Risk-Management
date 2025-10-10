"""
作業目標 7.2: t分布參數估計 (t-Distribution Fitting)

背景:
t分布在金融建模中比常態分布更適合描述金融回報，
因為它具有厚尾特性，能更好地捕捉極端事件。

問題:
如何估計t分布的參數？

解法 - 最大概似估計法:
t分布有三個參數：
- ν (nu): 自由度參數，控制厚尾程度
- μ (mu): 位置參數（中心）
- σ (sigma): 尺度參數（分散程度）

分布特性:
密度函數：f(x) = Γ((ν+1)/2) / (√νπ Γ(ν/2)) * (1 + (x-μ)²/(νσ²))^(-(ν+1)/2)

參數意義:
- ν → ∞: 收斂到常態分布
- ν < 4: 峰度無限大（超厚尾）
- ν = 1: 柯西分布（無限方差）

厚尾特性:
- P(|X| > k) 比常態分布大
- 更好描述金融市場的極端波動
- 考慮"黑天鵝"事件的影響

估計挑戰:
- 概似函數複雜，需要數值優化
- 參數間存在相關性
- 需要良好的初始值

應用優勢:
- 更準確的VaR估計
- 更保守的風險評估
- 更符合金融數據實際分布

模型檢驗:
- QQ圖比較
- Kolmogorov-Smirnov檢驗
- Anderson-Darling檢驗

實務考量:
- 樣本大小影響估計精度
- 異常值對估計的影響
- 時間變化的參數穩定性

風險管理應用:
t分布的厚尾特性使其在極端風險度量中表現更佳。
"""

import pandas as pd
import numpy as np
from library import T

if __name__ == "__main__":
    # 讀取數據進行t分布擬合
    data = pd.read_csv("../testfiles/data/test7_2.csv")["x1"].values
    
    # 使用library中的T類別進行最大概似估計
    t_model = T()
    t_model.fit(data)
    
    # 提取擬合的參數 [自由度, 位置, 尺度]
    nu = t_model.fitted_parameters[0]     # 自由度（厚尾程度）
    mu = t_model.fitted_parameters[1]     # 位置參數
    sigma = t_model.fitted_parameters[2]  # 尺度參數
    
    # 整理結果
    result = pd.DataFrame({
        "mu": [mu],
        "sigma": [sigma],
        "nu": [nu]
    })
    
    print(result)
