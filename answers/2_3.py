import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import ewCovar

if __name__ == "__main__":
    data = pd.read_csv("../testfiles/data/test2.csv")
    
    # EW Cov w/ EW Var(λ=0.94) EW Correlation(λ=0.97)
    cout = ewCovar(data.values, 0.97)
    sd1 = np.sqrt(np.diag(cout))
    cout = ewCovar(data.values, 0.94)
    sd = 1.0 / np.sqrt(np.diag(cout))
    result_matrix = np.diag(sd1) @ np.diag(sd) @ cout @ np.diag(sd) @ np.diag(sd1)
    
    result = pd.DataFrame(result_matrix, columns=data.columns, index=data.columns)
   
    print(result)