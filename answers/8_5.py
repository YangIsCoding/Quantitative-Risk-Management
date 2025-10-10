"""
作業目標 8.5: 期望損失計算 - t分布 (Expected Shortfall - t Distribution)

背景:
t分布的期望損失計算比常態分布更複雜，但能更好地捕捉
金融數據的厚尾特性，提供更準確的極端風險評估。

問題:
如何基於t分布假設計算期望損失？

解法 - t分布ES:
t分布的ES需要使用數值積分或近似公式計算

數學公式:
ES_α = μ + σ × (t_ν(t_α) × (ν + t_α²)) / ((ν - 1) × α)

其中:
- t_α: t分布的α分位數
- t_ν(t_α): t分布在t_α處的密度值
- ν: 自由度參數

t分布ES特性:
- 考慮厚尾分布的極端損失
- ES值通常比常態分布更大
- 更符合金融數據的實際風險

計算挑戰:
- 沒有封閉解析解
- 需要數值計算方法
- 參數估計的不確定性

與常態ES比較:
- t分布ES更保守
- 更好地反映極端事件
- 自由度越小，ES越大

實務應用:
- 更準確的資本配置
- 更可靠的壓力測試
- 更符合監管要求

數值實現:
使用scipy的積分函數或蒙地卡羅方法計算條件期望。

風險管理價值:
t分布ES為金融機構提供了更現實的極端損失評估，
特別適合高風險或新興市場投資。
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
    
    # 計算t分布的期望損失 (5%信心水準)
    alpha = 0.05
    
    # 方法1: 使用數值積分計算ES
    t_alpha = stats.t.ppf(alpha, df=nu, loc=mu, scale=sigma)
    
    # 對於標準化t分布的ES計算
    def es_t_standard(alpha, nu):
        """計算標準化t分布的ES"""
        t_alpha_std = stats.t.ppf(alpha, df=nu)
        density_at_quantile = stats.t.pdf(t_alpha_std, df=nu)
        es_std = -(density_at_quantile * (nu + t_alpha_std**2)) / ((nu - 1) * alpha)
        return es_std
    
    # 計算標準化ES然後轉換
    es_std_absolute = es_t_standard(alpha, nu)
    es_absolute = -(mu + sigma * es_std_absolute)
    
    # 相對ES (零中心化)
    es_std_relative = es_t_standard(alpha, nu)
    es_diff = -(sigma * es_std_relative)
    
    # 整理結果
    result = pd.DataFrame({
        'ES Absolute': [es_absolute],        # 絕對ES
        'ES Diff from Mean': [es_diff]      # 相對ES
    })
    
    print(result)