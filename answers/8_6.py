"""
作業目標 8.6: 蒙地卡羅模擬期望損失 (Monte Carlo Expected Shortfall)

背景:
蒙地卡羅模擬法計算ES特別適合複雜分布或無解析解的情況，
透過大量隨機抽樣估計條件期望損失。

問題:
如何使用蒙地卡羅模擬計算期望損失？

解法 - 模擬法ES:
1. 從擬合分布生成大量隨機樣本
2. 找出低於VaR閾值的樣本
3. 計算這些樣本的平均值

ES模擬步驟:
1. 生成N個隨機樣本
2. 計算第α×N個最小值作為VaR
3. ES = 所有小於VaR值的平均

數學表示:
ES_α = (1/(α×N)) × Σ(Xi), 其中 Xi ≤ VaR_α

優點分析:
相對於解析法：
- 適用於任何分布
- 容易實現和理解
- 可處理複雜情況

相對於參數法：
- 不需要複雜的公式
- 避免數值積分的困難
- 結果直觀易懂

模擬精度:
- 樣本數越多，估計越準確
- ES的標準誤約為 σ/√(αN)
- 需要足夠大的尾部樣本

技術實現:
- 使用與8.5相同的t分布參數
- 生成10,000個隨機樣本
- 計算經驗ES估計

實務考量:
- 計算成本與精度權衡
- 隨機數生成器品質
- 極值的統計特性

擴展應用:
- 投資組合ES計算
- 情境分析和壓力測試
- 複雜衍生品風險評估

模型驗證:
與解析法ES比較，驗證模擬準確性。

風險管理意義:
模擬法ES提供了靈活而直觀的極端風險評估工具。
"""

import numpy as np
import pandas as pd
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
    
    # 設定隨機種子確保結果可重現
    np.random.seed(42)
    
    # 從擬合的t分布生成10,000個隨機樣本
    simulated = stats.t.rvs(df=nu, loc=mu, scale=sigma, size=10000)
    
    # 計算5%期望損失 (ES)
    alpha = 0.05
    
    # 方法: 直接使用模擬樣本計算ES
    # 排序樣本以找到最小的α比例
    sorted_simulated = np.sort(simulated)
    cutoff_index = int(alpha * len(sorted_simulated))
    
    # ES = 最小α比例樣本的平均值
    tail_samples = sorted_simulated[:cutoff_index]
    es_absolute = -np.mean(tail_samples)
    
    # 相對ES (去除平均值影響)
    simulated_demean = simulated - np.mean(simulated)
    sorted_demean = np.sort(simulated_demean)
    tail_samples_demean = sorted_demean[:cutoff_index]
    es_diff = -np.mean(tail_samples_demean)
    
    # 整理結果
    result = pd.DataFrame({
        'ES Absolute': [es_absolute],        # 絕對ES
        'ES Diff from Mean': [es_diff]      # 相對ES
    })
    
    print(result)