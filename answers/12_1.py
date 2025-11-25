"""
題目 12.1: 歐式選擇權的布萊克-舒爾斯模型及希臘字母 (European Options GBSM with Greeks)

問題描述：
使用廣義布萊克-舒爾斯-墨頓（GBSM）模型計算歐式選擇權的理論價格
和風險敏感度指標（希臘字母）。希臘字母衡量選擇權價格對各種參數
變化的敏感度，是選擇權風險管理的核心工具。

目標：
1. 載入選擇權參數資料
2. 使用布萊克-舒爾斯模型計算選擇權價格
3. 計算希臘字母：Delta、Gamma、Vega、Rho、Theta
4. 輸出完整的選擇權評價結果

解法流程：
1. 讀取 test12_1.csv 檔案中的選擇權參數
2. 將到期天數轉換為年份
3. 對每個選擇權調用 Utils.bs_european_greeks()
4. 計算選擇權價格和所有希臘字母
5. 整理結果並輸出 DataFrame
"""

import pandas as pd
import library as Utils

# Test 12.1: European Options GBSM with Greeks
data_path = "../testfiles/data/test12_1.csv"
options_data = pd.read_csv(data_path)

# Filter out empty rows
options_data = options_data.dropna(subset=['ID'])

results = []

for _, option in options_data.iterrows():
    # Convert days to years
    T = option['DaysToMaturity'] / option['DayPerYear']
    
    # Calculate option price and Greeks using Black-Scholes
    result = Utils.bs_european_greeks(
        S=option['Underlying'],
        K=option['Strike'],
        T=T,
        r=option['RiskFreeRate'],
        q=option['DividendRate'],
        sigma=option['ImpliedVol'],
        option_type=option['Option Type'].lower()
    )
    
    # Add ID to result
    result['ID'] = option['ID']
    results.append(result)

# Convert to DataFrame and reorder columns
results_df = pd.DataFrame(results)
results_df = results_df[['ID', 'Price', 'Delta', 'Gamma', 'Vega', 'Rho', 'Theta']]

# Rename Price column to Value for consistency
results_df = results_df.rename(columns={'Price': 'Value'})

print(results_df)