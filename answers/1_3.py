import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import missing_cov

if __name__ == "__main__":
    data = pd.read_csv("../testfiles/data/test1.csv")
    
    # Pairwise - Covariance
    cov_matrix = missing_cov(data.values, skipMiss=False)
    
    result = pd.DataFrame(cov_matrix, columns=data.columns, index=data.columns)
    
    result.to_csv("../testfiles/data/testout_1.3.csv", index=False)
    print(result)