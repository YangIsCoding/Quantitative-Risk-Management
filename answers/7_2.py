import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize

def fit_t_distribution(data):

    mu0 = np.mean(data)
    sigma0 = np.std(data)
    nu0 = 5.0
    init_params = [mu0, sigma0, nu0]

    def neg_log_likelihood(params):
        mu, sigma, nu = params
        return -np.sum(stats.t.logpdf(data, df=nu, loc=mu, scale=sigma))

    bounds = [(None, None), (1e-6, None), (2.1, None)]
    result = minimize(neg_log_likelihood, init_params, bounds=bounds, method="L-BFGS-B")

    mu_hat, sigma_hat, nu_hat = result.x
    return mu_hat, sigma_hat, nu_hat



if __name__ == "__main__":

    data = pd.read_csv("../testfiles/data/test7_2.csv")["x1"].values
    

    mu_hat, sigma_hat, nu_hat = fit_t_distribution(data)
    

    result = pd.DataFrame({
        "mu": [mu_hat],
        "sigma": [sigma_hat],
        "nu": [nu_hat]
    })

    pd.set_option('display.float_format', '{:.15f}'.format)
    print(result)
