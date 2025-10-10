"""
作業目標 6.1: 離散回報率計算 (Arithmetic Returns)

背景:
回報率是金融分析的基礎，有多種計算方式。
離散回報率（算術回報率）是最直觀的方式。

問題:
如何從價格時間序列計算離散回報率？

解法 - 離散回報率公式:
rt = (Pt - Pt-1) / Pt-1 = Pt/Pt-1 - 1

數學特性:
- 直觀易懂：直接反映價格變化百分比
- 加法性：投資組合回報 = 權重加權平均
- 時間不可加：多期回報 ≠ 單期回報之和

計算步驟:
1. 讀取價格數據（包含日期列）
2. 按時間順序排列
3. 計算相鄰期間的價格比值
4. 減1得到回報率

參數說明:
- method="DISCRETE": 指定離散回報率計算
- dateColumn="Date": 指定日期欄位名稱

應用場景:
- 投資組合權重計算
- 資產配置分析
- 單期績效評估
- 風險預算分配

實務考量:
- 適合短期分析
- 便於權重加權計算
- 直觀的百分比解釋

數據處理:
- 自動處理缺失值
- 保持原有資產順序
- 輸出帶日期的時間序列

統計意義:
離散回報率保持了價格變化的原始比例關係，
適合分析短期投資表現。
"""

import pandas as pd
import numpy as np
from library import return_calculate

if __name__ == "__main__":
    # 讀取價格數據
    prices = pd.read_csv("../testfiles/data/test6.csv")
    
    # 計算離散回報率（算術回報率）
    # rt = (Pt - Pt-1) / Pt-1
    returns = return_calculate(prices, method="DISCRETE", dateColumn="Date")
    
    # 保存結果到CSV文件
    returns.to_csv("../testfiles/data/testout6_1.csv", index=False)
    print(returns.head())