import pandas as pd
from scipy import stats

fitted_params = pd.read_csv("../testfiles/data/testout7_2.csv")
mu = fitted_params['mu'].iloc[0]
sigma = fitted_params['sigma'].iloc[0]
nu = fitted_params['nu'].iloc[0]

var_absolute = stats.t.ppf(0.05, df=nu, loc=mu, scale=sigma)
var_diff = stats.t.ppf(0.05, df=nu, loc=0, scale=sigma)

result = pd.DataFrame({
    'VaR Absolute': [-var_absolute],
    'VaR Diff from Mean': [-var_diff]
})

print(result)