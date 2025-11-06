"""
作業目標 12.2: 美式選擇權二元樹定價與希臘字母計算

背景:
美式選擇權可在到期前任何時間執行，無法使用封閉解析解，
二元樹方法提供了數值解法，能處理提前執行的最適策略。

問題:
如何使用二元樹方法計算美式選擇權價格和希臘字母？

解法 - 二元樹數值法:
1. 建構標的資產價格樹
2. 向後遞推計算選擇權價值
3. 在每節點比較持有vs執行價值
4. 使用數值微分計算希臘字母

二元樹建構:
標的價格演變: S(i,j) = S₀ × u^(i-j) × d^j
其中:
- u = e^(σ√Δt): 上漲因子
- d = 1/u = e^(-σ√Δt): 下跌因子
- Δt = T/N: 時間間隔

風險中性機率:
p = (e^(cΔt) - d) / (u - d)
其中 c 為持有成本率

美式選擇權遞推:
在每個節點計算:
1. 歐式價值: e^(-rΔt) × [p×V_up + (1-p)×V_down]
2. 執行價值: max(S-K, 0) 或 max(K-S, 0)
3. 美式價值: max(歐式價值, 執行價值)

希臘字母數值計算:
- Delta: (V(S+ΔS) - V(S-ΔS)) / (2ΔS)
- Gamma: (V(S+ΔS) + V(S-ΔS) - 2V(S)) / (ΔS)²
- Vega: (V(σ+Δσ) - V(σ)) / Δσ
- Rho: (V(r+Δr) - V(r)) / Δr
- Theta: (V(T-ΔT) - V(T)) / (-ΔT)

數值穩定性:
- 足夠的時間步數(N≥500)確保收斂
- 適當的擾動大小避免數值誤差
- 中央差分法提高精確度

美式vs歐式特徵:
- 美式選擇權價值≥歐式選擇權價值
- 提前執行邊界的計算
- 對利率和股利敏感度更高

計算流程:
1. 讀取選擇權參數
2. 設定二元樹參數(步數、擾動量)
3. 計算基準選擇權價格
4. 透過參數擾動計算希臘字母
5. 輸出完整風險指標

技術考量:
- 樹步數與計算精度權衡
- 數值微分擾動量選擇
- 記憶體使用最佳化

實務應用:
- 美式選擇權交易策略
- 提前執行時機分析
- 複雜衍生品定價
- 風險管理系統

模型優勢:
- 處理複雜邊界條件
- 靈活的付息結構
- 直觀的計算過程
- 容易擴展和修改

模型限制:
- 計算成本較高
- 需要足夠步數確保精度
- 數值誤差累積問題
"""

# Problem 12.2: American Options with Binary Tree including Greeks
# 使用library.py中的函數計算美式選擇權的價格和希臘字母

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
        
        # 判斷是否為買權
        is_call = (option['Option Type'] == 'Call')
        
        # 計算美式選擇權價格和希臘字母
        result = calculate_american_greeks_numerical(
            is_call,
            option['Underlying'],
            option['Strike'],
            ttm,
            option['RiskFreeRate'],
            c,
            option['ImpliedVol'],
            N=500
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