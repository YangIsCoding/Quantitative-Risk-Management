"""
作業目標 2.1: 指數加權協方差矩陣計算 (EWMA Covariance)

背景:
傳統樣本協方差給予所有觀察值相等權重，但在金融時間序列中，近期觀察值通常比遠期更重要。
指數加權移動平均(EWMA)給予近期數據更高權重，能更好地反映當前市場狀況。

問題:
如何計算能反映近期市場波動的協方差矩陣？

解法 - 指數加權協方差矩陣(EWMA):
使用衰減因子λ給予不同時間點的觀察值不同權重。

數學公式:
σt² = λσt-1² + (1-λ)rt-1²
權重: wi = (1-λ)λ^(n-i-1) for i = 0,1,...,n-1

參數說明:
λ = 0.97: RiskMetrics建議的日頻數據衰減因子
- λ越大：記憶越長，變化越平滑
- λ越小：對近期數據反應越敏感

應用:
- VaR模型中的波動率預測
- 投資組合風險管理
- 衍生品定價中的動態避險

優點:
1. 對近期市場變化反應迅速
2. 計算效率高（遞歸更新）
3. 業界標準方法
"""

import pandas as pd
import numpy as np
from library import ewCovar

if __name__ == "__main__":
    # 讀取測試數據（40個觀察值，5個變數）
    data = pd.read_csv("../testfiles/data/test2.csv")
    
    # 計算指數加權協方差矩陣，λ=0.97
    # 0.97是RiskMetrics建議的日頻數據標準參數
    cov_matrix = ewCovar(data.values, 0.97)
    
    # 將結果轉換為帶有變數名稱的DataFrame
    result = pd.DataFrame(cov_matrix, columns=data.columns, index=data.columns)
    
    print(result)