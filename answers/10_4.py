import library as Utils
import pandas as pd

# Test10.4: Maximum Sharpe Ratio Portfolio with Bounds Constraints
cov = pd.read_csv("../testfiles/data/test5_2.csv").values
mu = pd.read_csv("../testfiles/data/test10_3_means.csv").values.flatten()

# Maximum Sharpe ratio optimization with rf = 0.04 and bounds [0.1, 0.5] for all assets
# Using the max_sharpe_ratio function from library with bounds constraints
weights, sharpe_ratio = Utils.max_sharpe_ratio(mu, cov, rf=0.04, bounds=[(0.1, 0.5)] * 5, long_only=True)
print(weights)