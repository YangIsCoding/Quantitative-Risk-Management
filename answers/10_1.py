import library as Utils
import pandas as pd

# Test10.1: Basic Risk Parity Portfolio (Equal Risk Budget)
cov = pd.read_csv("../testfiles/data/test5_2.csv").values

# Using the risk_parity_csd function from library
weights = Utils.risk_parity_csd(cov)
print(weights)