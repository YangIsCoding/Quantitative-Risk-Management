import pandas as pd
from scipy import stats

def fit_normal(data):
    mu, sigma = stats.norm.fit(data)
    return mu, sigma


data = pd.read_csv('../testfiles/data/test7_1.csv')
x_values = data['x1'].values


mu_hat, sigma_hat = fit_normal(x_values)

result = pd.DataFrame({
    'mu': [mu_hat],
    'sigma': [sigma_hat]
})

pd.set_option('display.float_format', '{:.18f}'.format)
print(result)