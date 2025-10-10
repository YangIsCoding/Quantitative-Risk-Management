"""
作業目標 8.2: t分布風險值計算 (t-Distribution VaR)

背景:
相對於常態分布，t分布具有厚尾特性，能更好地捕捉金融市場的極端風險，
提供更保守和現實的VaR估計。

問題:
如何基於t分布假設計算VaR？

解法 - t分布參數法VaR:
使用第7.2題擬合的t分布參數計算分位數

VaR計算公式:
VaR_α = -F_t^(-1)(α; ν, μ, σ)
其中 α = 0.05 (5%機率)

t分布參數:
- ν (nu): 自由度，控制厚尾程度
- μ (mu): 位置參數
- σ (sigma): 尺度參數

厚尾效應:
- ν越小，尾部越厚
- 極端事件機率較常態分布高
- VaR值通常較常態分布大

與常態VaR的比較:
相同點：
- 同樣計算5%分位數
- 同樣區分絕對和相對VaR

差異點：
- t分布考慮厚尾特性
- 提供更保守的風險估計
- 更適合金融數據特性

實務優勢:
- 更準確的極端風險評估
- 符合監管機構的保守原則
- 減少VaR突破次數

統計理論:
當ν→∞時，t分布收斂到常態分布
當ν較小時，顯著偏離常態分布

風險管理意義:
- 更好的資本配置
- 更準確的風險預算
- 更可靠的壓力測試

模型驗證:
- 回測分析（突破頻率）
- 與歷史模擬法比較
- Kupiec檢驗

應用場景:
- 高波動市場期間
- 新興市場投資
- 衍生品風險管理
- 另類投資策略

計算考量:
需要數值方法計算t分布的分位數函數。
"""

import pandas as pd
import numpy as np
from scipy import stats
from library import T

if __name__ == "__main__":
    # 讀取數據，使用第7.2題相同的數據
    data = pd.read_csv("../testfiles/data/test7_2.csv")["x1"].values
    
    # 使用library擬合t分布
    t_model = T()
    t_model.fit(data)
    
    # 提取擬合參數
    nu = t_model.fitted_parameters[0]     # 自由度
    mu = t_model.fitted_parameters[1]     # 位置參數
    sigma = t_model.fitted_parameters[2]  # 尺度參數
    
    # 計算t分布VaR (5%信心水準)
    # 使用負號因為VaR表示損失（正值）
    var_absolute = -stats.t.ppf(0.05, df=nu, loc=mu, scale=sigma)
    var_diff = -stats.t.ppf(0.05, df=nu, loc=0, scale=sigma)
    
    # 整理結果
    result = pd.DataFrame({
        'VaR Absolute': [var_absolute],        # 絕對VaR
        'VaR Diff from Mean': [var_diff]      # 相對VaR
    })
    
    print(result)