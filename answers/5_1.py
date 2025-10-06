import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import near_psd

# 讀取共變異數矩陣（不設定index_col，保持5x5格式）
cov_matrix = pd.read_csv("../testfiles/data/test5_1.csv")

# 移除第一列（列名），只保留數值
cov_values = cov_matrix.values

# 應用 near_psd 函數 
result_matrix = near_psd(cov_values)

# 轉換為 DataFrame 格式，移除第一列
result_df = pd.DataFrame(result_matrix[:, 1:], 
                        index=['x1', 'x2', 'x3', 'x4', 'x5'],
                        columns=['x2', 'x3', 'x4', 'x5'])

print(result_df)