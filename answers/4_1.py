import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import chol_psd

if __name__ == "__main__":
    # Cholesky factorization - read from testout_3.1.csv
    cin = pd.read_csv("../testfiles/data/testout_3.1.csv").values
    n, m = cin.shape
    cout = np.zeros((n, m))
    
    # Custom chol_psd implementation
    try:
        chol_matrix = np.linalg.cholesky(cin)
        cout = chol_matrix
    except:
        cout = chol_psd(cin)
    
    result = pd.DataFrame(cout)
    
    print(result)