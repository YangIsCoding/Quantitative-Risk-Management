import library as Utils
import pandas as pd

# Test 11.2: Factor Attribution Analysis
asset_returns = pd.read_csv("../testfiles/data/test11_2_stock_returns.csv")
betas = pd.read_csv("../testfiles/data/test11_2_beta.csv", index_col=0)
init_weights = pd.read_csv("../testfiles/data/test11_2_weights.csv").values.flatten()
factor_returns = pd.read_csv("../testfiles/data/test11_2_factor_returns.csv")

# Using the factor_attribution function from library
result = Utils.factor_attribution(asset_returns, betas, init_weights, factor_returns)
print(result)