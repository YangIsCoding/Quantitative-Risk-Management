import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import ewCovar

if __name__ == "__main__":
    data = pd.read_csv("../testfiles/data/test2.csv")
    
    # EW Correlation λ=0.94
    cout = ewCovar(data.values, 0.94)
    sd = 1.0 / np.sqrt(np.diag(cout))
    corr_matrix = np.diag(sd) @ cout @ np.diag(sd)
    
    result = pd.DataFrame(corr_matrix, columns=data.columns, index=data.columns)
    
    print(result)