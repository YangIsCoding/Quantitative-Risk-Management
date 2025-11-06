"""
作業目標 12.1: 歐式選擇權GBSM定價與希臘字母計算

背景:
Black-Scholes-Merton (GBSM) 模型是選擇權定價的經典理論，
提供歐式選擇權的封閉解析解，並可計算風險敏感度指標。

問題:
如何使用GBSM模型計算歐式選擇權價格和希臘字母？

解法 - GBSM解析法:
1. 使用Black-Scholes公式計算選擇權價格
2. 計算各種希臘字母衡量風險敏感度
3. 支援連續股利和持有成本模型

GBSM核心公式:
對於買權: C = S×e^((c-r)T)×N(d1) - K×e^(-rT)×N(d2)
對於賣權: P = K×e^(-rT)×N(-d2) - S×e^((c-r)T)×N(-d1)

其中:
d1 = [ln(S/K) + (c + σ²/2)T] / (σ√T)
d2 = d1 - σ√T

希臘字母計算:
- Delta (Δ): 標的價格敏感度 ∂V/∂S
- Gamma (Γ): Delta變化敏感度 ∂²V/∂S²
- Vega (ν): 波動率敏感度 ∂V/∂σ
- Theta (Θ): 時間衰減敏感度 ∂V/∂t
- Rho (ρ): 利率敏感度 ∂V/∂r

持有成本模型:
c = r - q (Merton 1973): 連續股利率
c = 0 (Black 1976): 期貨選擇權
c = r - r_f (Garman-Kohlhagen 1983): 匯率選擇權

計算流程:
1. 讀取選擇權參數資料
2. 計算持有成本 c = r - div_yield
3. 使用GBSM公式計算價格
4. 計算所有希臘字母
5. 輸出完整風險報告

技術實現:
- 使用scipy.stats.norm計算累積分佈函數
- 支援買權和賣權計算
- 處理不同到期時間和履約價格

風險管理意義:
希臘字母提供選擇權交易的風險控制工具：
- Delta: 避險比例計算
- Gamma: 動態避險頻率
- Vega: 波動率風險管理
- Theta: 時間價值衰減評估
- Rho: 利率風險暴露

實務應用:
- 選擇權做市商避險策略
- 投資組合風險管理
- 衍生品定價驗證
- 監管資本計提

模型限制:
- 假設常數波動率和利率
- 不適用於美式選擇權
- 忽略交易成本和流動性風險
"""

# Problem 12.1: European Options GBSM with Greeks
# 使用library.py中的函數計算歐式選擇權的價格和希臘字母

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import *
import pandas as pd
import numpy as np

def main():
    # 讀取測試資料
    data_path = "../testfiles/data/test12_1.csv"
    options_data = pd.read_csv(data_path)
    
    # 過濾掉空白行
    options_data = options_data.dropna(subset=['ID'])
    
    results = []
    
    for _, option in options_data.iterrows():
        # 計算持有成本 (carry rate)
        c = option['RiskFreeRate'] - option['DividendRate']
        
        # 轉換到期時間為年
        ttm = option['DaysToMaturity'] / option['DayPerYear']
        
        # 計算選擇權價格和希臘字母
        result = gbsm_with_greeks(
            option['Option Type'],
            option['Underlying'],
            option['Strike'],
            ttm,
            option['RiskFreeRate'],
            c,
            option['ImpliedVol']
        )
        
        # 添加ID到結果中
        result['ID'] = option['ID']
        results.append(result)
    
    # 轉換為DataFrame並重新排序列
    results_df = pd.DataFrame(results)
    results_df = results_df[['ID', 'Value', 'Delta', 'Gamma', 'Vega', 'Rho', 'Theta']]
    
    print(results_df)
    
    return results_df

if __name__ == "__main__":
    main()