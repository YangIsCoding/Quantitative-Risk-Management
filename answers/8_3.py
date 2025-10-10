"""
作業目標 8.3: 蒙地卡羅模擬風險值 (Monte Carlo VaR)

背景:
蒙地卡羅模擬法是計算VaR的重要方法，特別適合複雜分布或
無法解析求解分位數的情況。

問題:
如何使用蒙地卡羅模擬計算VaR？

解法 - 模擬法VaR:
1. 使用擬合的t分布參數
2. 生成大量隨機樣本
3. 計算樣本的經驗分位數

模擬步驟:
1. 參數擬合：使用第7.2題的t分布參數
2. 隨機抽樣：生成10,000個t分布隨機數
3. 分位數計算：計算5%經驗分位數

優點分析:
相對於解析法：
- 不需要複雜的分位數函數
- 適用於任意分布形式
- 容易處理組合風險

相對於歷史模擬：
- 使用完整的分布信息
- 可以外推超出歷史數據的情況
- 參數化便於情境分析

技術實現:
- 設定隨機種子確保可重現性
- 使用足夠大的樣本數（10,000）
- 經驗分位數估計

抽樣精度:
VaR估計的標準誤約為 √(p(1-p)/n) = √(0.05×0.95/10000) ≈ 0.0022
其中p=0.05, n=10000

與解析VaR比較:
- 理論上應該收斂到8.2的結果
- 實際存在蒙地卡羅誤差
- 樣本數越大，精度越高

擴展應用:
- 投資組合VaR
- 路徑相依衍生品
- 複雜相關結構
- 壓力測試情境

實務考量:
- 計算成本與精度權衡
- 隨機數品質的影響
- 方差減少技術

模型驗證:
與解析法結果比較，驗證模擬精度。

風險管理價值:
模擬法提供了VaR計算的靈活性和可擴展性。
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
    
    # 使用經驗分位數計算VaR (5%分位數)
    var_absolute = -np.percentile(simulated, 5)
    var_diff = -np.percentile(simulated - np.mean(simulated), 5)
    
    # 整理結果
    result = pd.DataFrame({
        'VaR Absolute': [var_absolute],        # 絕對VaR
        'VaR Diff from Mean': [var_diff]      # 相對VaR
    })
    
    print(result)