import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import higham_nearestPSD

if __name__ == "__main__":
    # Higham covariance - read from testout_1.3.csv
    cin = pd.read_csv("../testfiles/data/testout_1.3.csv")
    psd_matrix = higham_nearestPSD(cin.values)
    
    result = pd.DataFrame(psd_matrix, columns=cin.columns, index=cin.columns)
    
    print(result)