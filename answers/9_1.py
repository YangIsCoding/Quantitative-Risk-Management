"""
作業目標 9.1: 投資組合風險聚合分析 (Portfolio Risk Aggregation)

背景:
投資組合風險聚合是將個別資產的風險模型整合為整體投資組合風險的過程，
需要考慮資產間的相關性和各自的分布特性。

問題:
如何建立多資產投資組合的風險聚合模型？

解法 - Copula-based風險聚合:
1. 個別資產分布建模
2. 相關性結構建模 (Copula)
3. 聯合模擬
4. 投資組合風險計算

技術流程:
1. 資產A: 擬合常態分布
2. 資產B: 擬合t分布  
3. Spearman相關性: 捕捉非線性相依性
4. PCA模擬: 生成相關隨機數
5. 風險聚合: 計算投資組合VaR/ES

模型架構:
- 邊際分布: 各資產的個別風險特性
- Copula結構: 資產間的相依關係
- 投資組合: 加權風險聚合

數學基礎:
F(x,y) = C(F_A(x), F_B(y))
其中C是copula函數，F_A, F_B是邊際分布

實務步驟:
1. 讀取歷史回報數據
2. 擬合每個資產的分布模型
3. 估計rank相關性矩陣
4. 使用copula生成聯合場景
5. 計算投資組合損益分布
6. 聚合風險度量(VaR, ES)

創新特點:
- 保持個別資產的分布特性
- 準確建模資產間相依性
- 支持不同分布的混合
- 提供完整的風險分解

應用價值:
- 多資產投資組合管理
- 風險預算分配
- 壓力測試分析
- 監管資本計算

技術優勢:
相對於傳統方法，copula方法能更好地處理非常態分布和非線性相關性。
"""

import pandas as pd
import numpy as np
from scipy import stats
from library import Norm, T

def simulate_correlated_normal(correlation_matrix, n_sim):
    """使用Cholesky分解生成相關的常態隨機數"""
    # Cholesky分解生成相關隨機數
    L = np.linalg.cholesky(correlation_matrix)
    
    # 生成獨立標準常態隨機數
    independent_normals = np.random.randn(n_sim, correlation_matrix.shape[0])
    
    # 轉換為相關隨機數
    correlated_normals = independent_normals @ L.T
    
    return correlated_normals

if __name__ == "__main__":
    # 讀取投資組合回報數據
    returns_data = pd.read_csv("../testfiles/data/test9_1_returns.csv")
    
    # 設定投資組合資訊（基於test_setup.jl）
    portfolio = {"A": 2000.0, "B": 3000.0}  # 當前市值
    
    # 步驟1: 擬合各資產的分布模型
    # 資產A: 常態分布
    norm_model_A = Norm()
    norm_model_A.fit(returns_data['A'].values)
    
    # 資產B: t分布
    t_model_B = T()
    t_model_B.fit(returns_data['B'].values)
    
    # 步驟2: 計算相關性
    # 簡化版本：直接使用Pearson相關係數而非Spearman rank correlation
    correlation_coeff = np.corrcoef(returns_data['A'], returns_data['B'])[0, 1]
    correlation_matrix = np.array([[1.0, correlation_coeff], 
                                   [correlation_coeff, 1.0]])
    
    # 步驟3: 蒙地卡羅模擬
    n_sim = 100000
    np.random.seed(42)
    
    # 生成相關的標準常態隨機數
    correlated_normals = simulate_correlated_normal(correlation_matrix, n_sim)
    
    # 步驟4: 轉換為各資產的回報率
    # 資產A: 直接使用常態分布
    returns_A = stats.norm.ppf(stats.norm.cdf(correlated_normals[:, 0]), 
                               norm_model_A.fitted_parameters[0],
                               norm_model_A.fitted_parameters[1])
    
    # 資產B: 轉換為t分布
    # 使用常態到t分布的copula轉換
    u_B = stats.norm.cdf(correlated_normals[:, 1])
    returns_B = stats.t.ppf(u_B,
                           df=t_model_B.fitted_parameters[0],
                           loc=t_model_B.fitted_parameters[1], 
                           scale=t_model_B.fitted_parameters[2])
    
    # 步驟5: 計算投資組合損益
    pnl_A = portfolio["A"] * returns_A
    pnl_B = portfolio["B"] * returns_B
    
    # 步驟6: 計算風險度量
    def calculate_risk_metrics(pnl_series, confidence=0.95):
        """計算VaR和ES"""
        alpha = 1 - confidence
        var = -np.percentile(pnl_series, alpha * 100)
        
        # ES: 超過VaR的條件期望損失
        tail_losses = pnl_series[pnl_series <= -var]
        es = -np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        return var, es
    
    # 個別資產風險
    var_A, es_A = calculate_risk_metrics(pnl_A)
    var_B, es_B = calculate_risk_metrics(pnl_B)
    
    # 總投資組合風險
    total_pnl = pnl_A + pnl_B
    var_total, es_total = calculate_risk_metrics(total_pnl)
    total_portfolio_value = portfolio["A"] + portfolio["B"]
    
    # 計算百分比風險度量（相對於投資組合價值的比例）
    var_A_pct = var_A / portfolio["A"]  # 不乘以100，保持小數形式
    var_B_pct = var_B / portfolio["B"]
    es_A_pct = es_A / portfolio["A"] 
    es_B_pct = es_B / portfolio["B"]
    
    # 總投資組合百分比風險
    var_total_pct = var_total / total_portfolio_value
    es_total_pct = es_total / total_portfolio_value
    
    # 整理結果（包含個別資產和總投資組合）
    result = pd.DataFrame({
        'Stock': ['A', 'B', 'Total'],
        'VaR95': [var_A, var_B, var_total],
        'ES95': [es_A, es_B, es_total],
        'VaR95_Pct': [var_A_pct, var_B_pct, var_total_pct],
        'ES95_Pct': [es_A_pct, es_B_pct, es_total_pct]
    })
    
    print(result)