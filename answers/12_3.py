"""
作業目標 12.3: 含離散股利的美式選擇權定價

背景:
實務中多數股票支付離散股利而非連續股利，
股利發放會影響標的價格路徑，需要特殊處理方法。

問題:
如何在二元樹模型中處理離散股利發放？

解法 - 修正二元樹法:
1. 建立股利調整後的價格樹
2. 在股利發放日扣除股利金額
3. 確保價格樹的數學一致性
4. 計算美式選擇權最適執行策略

股利處理機制:
傳統方法問題：
- 直接扣除股利破壞價格樹結構
- 重組性假設不成立
- 數值不穩定

解決方案：
1. 股利發放日價格調整
2. 樹結構重新校準
3. 非重組樹建構

股利調整公式:
除息日前: S(t) = 股價含息
除息日後: S(t) = 股價含息 - 股利金額
確保: S_adjusted ≥ 0

二元樹修正:
標準樹: S(i,j) = S₀ × u^(i-j) × d^j

股利樹: S(i,j) = (S₀ - PV(Dividends)) × u^(i-j) × d^j + PV(Remaining_Dividends)

其中 PV 為股利現值

離散股利影響:
1. 買權價值: 股利降低標的價格，減少買權價值
2. 賣權價值: 股利增加賣權價值
3. 提前執行: 
   - 買權：股利前可能執行
   - 賣權：股利後執行價值可能提高

計算複雜度:
- 每個股利日期需要特殊處理
- 樹節點數量指數增長
- 記憶體需求大幅增加

數值穩定技巧:
1. 股利金額不能超過股價
2. 負股價處理：設為零
3. 風險中性機率檢查
4. 收斂性測試

實務考量:
股利預測準確性:
- 歷史股利模式
- 公司財務政策
- 市場環境變化

執行策略:
- 股利前執行買權評估
- 美式賣權最適時機
- 避險策略調整

技術實現:
1. 解析股利日期字串
2. 轉換為樹步數格式
3. 逐步扣除股利影響
4. 向後遞推計算價值

模型驗證:
- 與連續股利模型比較
- 極限情況測試
- 歷史數據回測

金融市場應用:
- 股票選擇權定價
- 股利避險策略
- 企業行動影響評估
- 投資組合風險管理

模型限制:
- 股利金額預測誤差
- 除息日價格跳躍簡化
- 交易成本忽略
- 流動性影響未考慮

擴展應用:
- 多次股利發放
- 股票分割處理
- 特殊股利影響
- 複合企業行動
"""

# Problem 12.3: American Options with Discrete Dividends
# 使用library.py中的函數計算含離散股利的美式選擇權價格

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import *
import pandas as pd
import numpy as np

def main():
    # 讀取測試資料
    data_path = "../testfiles/data/test12_3.csv"
    options_data = pd.read_csv(data_path)
    
    # 過濾掉空白行
    options_data = options_data.dropna(subset=['ID'])
    
    results = []
    
    for _, option in options_data.iterrows():
        # 轉換到期時間為年
        ttm = option['DaysToMaturity'] / 365
        
        # 解析股利日期和金額
        div_dates_str = option['DividendDates'].split(',')
        div_amounts_str = option['DividendAmts'].split(',')
        
        div_dates = [int(d.strip()) for d in div_dates_str]
        div_amounts = [float(a.strip()) for a in div_amounts_str]
        
        # 計算二元樹步數（按照測試設定，每天2步）
        N = option['DaysToMaturity'] * 2
        
        # 調整股利日期到二元樹步數
        div_dates_adjusted = [d * 2 for d in div_dates]
        
        # 判斷是否為買權
        is_call = (option['Option Type'] == 'Call')
        
        # 計算含離散股利的美式選擇權價格
        option_value = binomial_american_dividend(
            is_call,
            option['Underlying'],
            option['Strike'],
            ttm,
            option['RiskFreeRate'],
            div_dates_adjusted,
            div_amounts,
            option['ImpliedVol'],
            N
        )
        
        results.append({
            'ID': option['ID'],
            'Value': option_value
        })
    
    # 轉換為DataFrame
    results_df = pd.DataFrame(results)
    
    print(results_df)

    
    return results_df

if __name__ == "__main__":
    main()