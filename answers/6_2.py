import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from library import return_calculate

if __name__ == "__main__":
    # Log returns
    prices = pd.read_csv("../testfiles/data/test6.csv")
    returns = return_calculate(prices, method="LOG", dateColumn="Date")
    
    print(returns.head())