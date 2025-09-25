import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import return_calculate

if __name__ == "__main__":
    # Arithmetic returns
    prices = pd.read_csv("../testfiles/data/test6.csv")
    returns = return_calculate(prices, method="DISCRETE", dateColumn="Date")
    
    returns.to_csv("../testfiles/data/testout6_1.csv", index=False)
    print(returns.head())