import numpy as np
import pandas as pd
from scipy import stats

fitted_params = pd.read_csv("../testfiles/data/testout7_2.csv")
mu = fitted_params['mu'].iloc[0]
sigma = fitted_params['sigma'].iloc[0]
nu = fitted_params['nu'].iloc[0]

np.random.seed(42)
simulated = stats.t.rvs(df=nu, loc=mu, scale=sigma, size=10000)

var_absolute = np.percentile(simulated, 5)
var_diff = np.percentile(simulated - np.mean(simulated), 5)

result = pd.DataFrame({
    'VaR Absolute': [-var_absolute],
    'VaR Diff from Mean': [-var_diff]
})

print(result)