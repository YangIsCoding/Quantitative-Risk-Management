import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import higham_nearestPSD

if __name__ == "__main__":
    # Higham Correlation - read from testout_1.4.csv
    cin = pd.read_csv("../testfiles/data/testout_1.4.csv")
    psd_matrix = higham_nearestPSD(cin.values)
    
    result = pd.DataFrame(psd_matrix, columns=cin.columns, index=cin.columns)
    
    result.to_csv("../testfiles/data/testout_3.4.csv", index=False)
    print(result)