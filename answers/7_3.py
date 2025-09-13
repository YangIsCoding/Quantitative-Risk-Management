
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize


def nll(theta):
    beta = theta[:-2]
    sigma = theta[-2]
    nu = theta[-1]
    if sigma <= 0 or nu <= 2:
        return np.inf
    r = y - X @ beta
    ll = stats.t.logpdf(r, df=nu, loc=0.0, scale=sigma).sum()
    return -ll

cin = pd.read_csv("../testfiles/data/test7_3.csv")
y = cin["y"].values.astype(float)
X = cin.drop(columns=["y"]).values.astype(float)
X = np.column_stack([np.ones(len(X)), X]) 


beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta_ols
sigma0 = np.std(resid, ddof=X.shape[1])
nu0 = 5.0
theta0 = np.concatenate([beta_ols, [max(sigma0, 1e-3), nu0]])


p = X.shape[1]
bounds = [(None, None)] * p + [(1e-8, None), (2.0001, None)]

res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)


beta_hat = res.x[:-2]
sigma_hat = res.x[-2]
nu_hat = res.x[-1]
mu_hat = 0.0

out = pd.DataFrame({
    "mu":    [mu_hat],
    "sigma": [sigma_hat],
    "nu":    [nu_hat],
    "Alpha": [beta_hat[0]],
    "B1":    [beta_hat[1]],
    "B2":    [beta_hat[2]],
    "B3":    [beta_hat[3]],
})


pd.set_option('display.float_format', '{:.18f}'.format)
print(out)