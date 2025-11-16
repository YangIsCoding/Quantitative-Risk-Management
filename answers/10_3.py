import pandas as pd
import library as Utils

# Test10.3: Maximum Sharpe Ratio Portfolio
cov = pd.read_csv("../testfiles/data/test5_2.csv").values
mu = pd.read_csv("../testfiles/data/test10_3_means.csv").values.flatten()

# Using the max_sharpe_ratio function from library
weights, sharpe_ratio = Utils.max_sharpe_ratio(mu=mu, cov=cov, rf=0.04)
print(weights)