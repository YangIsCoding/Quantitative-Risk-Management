import library as Utils
import pandas as pd

# Test 11.1: Asset Attribution Analysis
returns = pd.read_csv("../testfiles/data/test11_1_returns.csv")
init_weights = pd.read_csv("../testfiles/data/test11_1_weights.csv").values.flatten()

# Using the asset_attribution function from library
result = Utils.asset_attribution(returns, init_weights)
print(result)