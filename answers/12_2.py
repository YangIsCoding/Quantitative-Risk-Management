"""
題目 12.2: 美式選擇權的二項樹模型及希臘字母 (American Options Binary Tree with Greeks)

問題描述：
使用二項樹方法計算美式選擇權的理論價格和希臘字母。美式選擇權
可以在到期日前任何時間執行，二項樹方法能夠處理這種路徑相依
的特性，並提供準確的評價結果。

目標：
1. 載入選擇權參數資料
2. 使用二項樹模型計算美式選擇權價格
3. 計算希臘字母：Delta、Gamma、Vega、Rho、Theta
4. 處理美式選擇權的提早執行特性

解法流程：
1. 讀取 test12_1.csv 檔案中的選擇權參數
2. 將到期天數轉換為年份
3. 對每個選擇權調用 Utils.american_binomial_with_greeks()
4. 使用 500 步的二項樹進行精確計算
5. 整理結果並輸出 DataFrame
"""

import pandas as pd
import library as Utils

# Test 12.2: American Options Binary Tree with Greeks
data_path = "../testfiles/data/test12_1.csv"
options_data = pd.read_csv(data_path)

# Filter out empty rows
options_data = options_data.dropna(subset=['ID'])

results = []

for _, option in options_data.iterrows():
    # Convert days to years
    T = option['DaysToMaturity'] / option['DayPerYear']
    
    # Calculate American option price and Greeks using binomial tree
    result = Utils.american_binomial_with_greeks(
        S=option['Underlying'],
        K=option['Strike'],
        T=T,
        r=option['RiskFreeRate'],
        q=option['DividendRate'],
        sigma=option['ImpliedVol'],
        steps=500,
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