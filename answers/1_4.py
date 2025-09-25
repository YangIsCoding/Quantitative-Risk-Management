import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import missing_cov

if __name__ == "__main__":
    data = pd.read_csv("../testfiles/data/test1.csv")
    
    corr_matrix = missing_cov(data.values, skipMiss=False, fun=np.corrcoef)
    
    result = pd.DataFrame(corr_matrix, columns=data.columns, index=data.columns)
    
    print(result)