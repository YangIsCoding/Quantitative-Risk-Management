import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import ewCovar

if __name__ == "__main__":
    data = pd.read_csv("../testfiles/data/test2.csv")
    
    # EW Covariance λ=0.97
    cov_matrix = ewCovar(data.values, 0.97)
    
    result = pd.DataFrame(cov_matrix, columns=data.columns, index=data.columns)
    
    print(result)