"""
作業目標 5.5: 主成分分析模擬 (PCA Simulation)

背景:
PCA模擬是降維模擬的重要技術，透過保留主要成分降低計算複雜度，
同時保持大部分的方差解釋能力。

問題:
如何使用主成分分析進行高效的多變量模擬？

解法 - PCA降維模擬法:
1. 特徵值分解：Σ = QΛQ'
2. 選擇成分：保留99%累積方差解釋度的主成分
3. 降維模擬：在主成分空間生成隨機數
4. 空間還原：轉換回原始變數空間

數學公式:
設原始協方差矩陣 Σ = QΛQ'
選擇前k個主成分滿足：Σᵢ₌₁ᵏ λᵢ / Σᵢ₌₁ⁿ λᵢ ≥ 0.99

模擬步驟:
1. Z ~ N(0,I_k) (k維獨立隨機數)
2. Y = Z√Λₖ (縮放到主成分方差)
3. X = YQₖ' (轉換回原始空間)

優點:
- 降低模擬維度（k < n）
- 計算效率大幅提升
- 保留主要風險因子
- 數值穩定性好

參數設定:
- 解釋度閾值：99%（平衡精度與效率）
- 樣本數：100,000
- 隨機種子：4

應用場景:
- 大型投資組合風險模擬
- 高維度因子模型
- 計算密集的風險度量
- 實時風險監控系統

效果評估:
比較PCA模擬與完整模擬的協方差矩陣差異，
驗證降維對風險結構的保持程度。

實務考量:
需要在計算效率與精度之間找到平衡點。
"""

import pandas as pd
import numpy as np

if __name__ == "__main__":
    # 讀取半正定協方差矩陣進行PCA模擬
    cin = pd.read_csv("../testfiles/data/test5_2.csv", header=None, skiprows=1).values.astype(float)
    
    # 步驟1: 特徵值分解
    eigenvals, eigenvecs = np.linalg.eigh(cin)
    
    # 步驟2: 按特徵值大小排序（從大到小）
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # 步驟3: 選擇解釋99%方差的主成分數量
    cumulative_explained = np.cumsum(eigenvals) / np.sum(eigenvals)
    n_components = np.argmax(cumulative_explained >= 0.99) + 1
    
    # 步驟4: 保留選定的主成分
    selected_eigenvals = eigenvals[:n_components]
    selected_eigenvecs = eigenvecs[:, :n_components]
    
    # 設定隨機種子確保結果可重現
    np.random.seed(4)
    
    # 步驟5: 在主成分空間生成隨機數
    Z = np.random.randn(100000, n_components)
    
    # 步驟6: 縮放到主成分方差
    scaled_Z = Z * np.sqrt(selected_eigenvals)
    
    # 步驟7: 轉換回原始變數空間
    simulated_data = scaled_Z @ selected_eigenvecs.T
    
    # 步驟8: 計算樣本協方差矩陣
    result_matrix = np.cov(simulated_data.T)
    
    result = pd.DataFrame(result_matrix)
    print(result)