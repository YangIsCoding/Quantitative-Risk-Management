"""
題目 12.3: 有離散股利的美式選擇權 (American Options with Discrete Dividends)

問題描述：
計算有離散股利支付的美式選擇權價格。實際市場中，股票常會支付
離散股利，這會影響選擇權的價值和最佳執行策略。需要使用修正的
二項樹方法來處理股利支付的影響。

目標：
1. 載入包含股利資訊的選擇權參數
2. 處理離散股利支付時點和金額
3. 使用修正二項樹計算美式選擇權價格
4. 考慮股利對選擇權執行策略的影響

解法流程：
1. 讀取 test12_3.csv 檔案中的選擇權和股利資料
2. 解析股利支付日期和金額
3. 將股利時點轉換為二項樹步數
4. 調用 Utils.bt_american_discrete_div() 計算選擇權價格
5. 輸出考慮股利影響後的選擇權價值
"""

import pandas as pd
import library as Utils

# Test 12.3: American Options with Discrete Dividends
data_path = "../testfiles/data/test12_3.csv"
options_data = pd.read_csv(data_path)

# Filter out empty rows
options_data = options_data.dropna(subset=['ID'])

results = []

for _, option in options_data.iterrows():
    # Convert days to years
    T = option['DaysToMaturity'] / 365
    
    # Parse dividend dates and amounts
    div_dates_str = option['DividendDates'].split(',')
    div_amounts_str = option['DividendAmts'].split(',')
    
    div_dates = [int(d.strip()) for d in div_dates_str]
    div_amounts = [float(a.strip()) for a in div_amounts_str]
    
    # Calculate binomial tree steps (2 steps per day as per test setting)
    N = option['DaysToMaturity'] * 2
    
    # Adjust dividend dates to binomial tree steps
    div_dates_adjusted = [d * 2 for d in div_dates]
    
    # Calculate American option with discrete dividends
    option_value = Utils.bt_american_discrete_div(
        S=option['Underlying'],
        K=option['Strike'],
        T=T,
        r=option['RiskFreeRate'],
        divAmts=div_amounts,
        divTimes=div_dates_adjusted,
        sigma=option['ImpliedVol'],
        steps=N,
        option_type=option['Option Type'].lower()
    )
    
    results.append({
        'ID': option['ID'],
        'Value': option_value
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

print(results_df)