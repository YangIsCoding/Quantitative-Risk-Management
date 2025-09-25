import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import near_psd

if __name__ == "__main__":
    # Near PSD Correlation - read from testout_1.4.csv
    cin = pd.read_csv("../testfiles/data/testout_1.4.csv")
    psd_matrix = near_psd(cin.values)
    
    result = pd.DataFrame(psd_matrix, columns=cin.columns, index=cin.columns)
    
    print(result)