import library as Utils
import pandas as pd
import numpy as np

# Test10.2: Risk Parity Portfolio with Custom Risk Budget
cov = pd.read_csv("../testfiles/data/test5_2.csv").values

# Custom risk budget: [2, 2, 2, 2, 1]
budget = np.array([2, 2, 2, 2, 1])

# Using the risk_parity_csd function from library with custom budget
weights = Utils.risk_parity_csd(cov, budget)
print(weights)