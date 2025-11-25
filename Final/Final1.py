

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings

from library import *

# problem 1
# a
problem1_data = pd.read_csv('problem1.csv')
returns = problem1_data['r'].values

S = 100
T = 1/255
r = 0.04 
q = 0

daily_vol = np.std(returns)
annual_vol = daily_vol * np.sqrt(255)

strikes = [99,100,101]

option_prices = []
for K in strikes:
    bs_result = bs_european_greeks(S, K, T, r, q, annual_vol, option_type='call')
    call_price = bs_result['Price']
    
    bs_result_put = bs_european_greeks(S, K, T, r, q, annual_vol, option_type='put')
    put_price = bs_result_put['Price']
    
    option_prices.append({'Strike': K, 'Call': call_price, 'Put': put_price})
    print(f"{K:6d} | {call_price:10.4f} | {put_price:9.4f}")

# b - implied vol

strikes_range = [95,96, 97, 98, 99, 100, 101, 102, 103, 104, 105]

implied_vol = []

for K in strikes_range:
    price = bs_european_greeks(S, K, T, r, q, annual_vol, option_type='call')['Price']
    iv = implied_volatility_solver(price, S, K, T, r, q, option_type='call', initial_guess=0.2)
    implied_vol.append(iv['implied_vol'])
        
plt.figure(figsize=(10, 6))
valid_strikes = [strike for strike, iv in zip(strikes_range, implied_vol) if not np.isnan(iv)]
valid_ivs = [iv for iv in implied_vol if not np.isnan(iv)]

plt.plot(valid_strikes, valid_ivs, 'b-', linewidth=2, label='implied volatility')
plt.xlabel('strike price')
plt.ylabel('implied volatility')
plt.title('implied volatility curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"iv range: {np.min(valid_ivs):.4f} 到 {np.max(valid_ivs):.4f}")
print()

# c
print("Theory would say this should be a flat line. Why is it not?")
print("In theory, under the BSM framework, the model assumes that the underlying asset's returns follow a log normal distribution with constant volatility.")
print("In reality, the volitilities could change rapidly, and it may not always be log normal distributio, it will have its own skew and kurtosis. That means, in reality, buyers will except to pay more for options that are OTM or ITM, leading to a volatility smile or skew.")

# problem 2
problem2_data = pd.read_csv("problem2.csv")
problem2_data['Date'] = pd.to_datetime(problem2_data['Date'])
spy_prices = problem2_data['SPY'].values

spy_returns = np.diff(spy_prices) / spy_prices[:-1]

current_spy_price = spy_prices[-1]
call_strike = 665
call_price = 7.05
put_strike = 655
put_price = 7.69
T_spy = 10 / 255
r_spy = 0.04
q_spy = 0.0109 

#a
mu_norm, sigma_norm = fit_normal(spy_returns)
loglik_norm = np.sum(stats.norm.logpdf(spy_returns, mu_norm, sigma_norm))
aic_norm = -2 * loglik_norm + 2 * 2


t_results = fit_t_distribution(spy_returns)

print(f"Normal Distribution:")
print(f"  μ = {mu_norm:.6f}, σ = {sigma_norm:.6f}")
print(f"  Log-likelihood = {loglik_norm:.2f}")
print(f"  AIC = {aic_norm:.2f}")
print("T-Distribution:")
print(f"  df = {t_results['df']:.3f}, loc = {t_results['loc']:.6f}, scale = {t_results['scale']:.6f}")
print(f"  Log-likelihood = {t_results['loglikelihood']:.2f}")
print(f"  AIC = {t_results['aic']:.2f}")

print("in this case, the t-distribution has a lovwer AIC, saying that t distribution is a better fit")

# b

call_iv = implied_volatility_solver(
    market_price=call_price,
    S=current_spy_price, K=call_strike, T=T_spy, r=r_spy, q=q_spy,
    option_type='call'
)

put_iv = implied_volatility_solver(
    market_price=put_price,
    S=current_spy_price, K=put_strike, T=T_spy, r=r_spy, q=q_spy,
    option_type='put'
)

print(f"  IV = {call_iv['implied_vol']:.4f} ({call_iv['implied_vol']*100:.2f}%)")
print(f"  IV = {put_iv['implied_vol']:.4f} ({put_iv['implied_vol']*100:.2f}%)")

# c 
T_holding = 5 / 255

n_simulations = 10000 

sim_returns = []
for _ in range(n_simulations):
    daily_returns = stats.t.rvs(
        df=t_results['df'], 
        loc=t_results['loc'], 
        scale=t_results['scale'], 
        size=5
    )
    cumulative_return = np.prod(1 + daily_returns) - 1
    sim_returns.append(cumulative_return)
sim_returns = np.array(sim_returns)

portfolio_values = []
for ret in sim_returns:
    future_spy_price = current_spy_price * (1 + ret)
    remaining_time = max(T_spy - T_holding, 0)
    
    if remaining_time > 0:
        call_value = bs_european_greeks(
            future_spy_price, call_strike, remaining_time, r_spy, q_spy,
            call_iv['implied_vol'], 'call'
        )['Price']
        
        put_value = bs_european_greeks(
            future_spy_price, put_strike, remaining_time, r_spy, q_spy,
            put_iv['implied_vol'], 'put'
        )['Price']
    else:
        call_value = max(future_spy_price - call_strike, 0)
        put_value = max(put_strike - future_spy_price, 0)
    
    portfolio_value = future_spy_price + put_value - call_value
    portfolio_values.append(portfolio_value)

portfolio_values = np.array(portfolio_values)

initial_portfolio_value = current_spy_price + put_price - call_price

portfolio_pnl = portfolio_values - initial_portfolio_value
portfolio_returns = portfolio_pnl / initial_portfolio_value

var_5pct = np.percentile(portfolio_returns, 5)
es_5pct = portfolio_returns[portfolio_returns <= var_5pct].mean()

print(f"  VaR (5%): {abs(var_5pct)*100:.2f}%")
print(f"  ES (5%): {abs(es_5pct)*100:.2f}%")
print(f"  VaR (5%): ${abs(var_5pct) * initial_portfolio_value:.2f}")
print(f"  ES (5%): ${abs(es_5pct) * initial_portfolio_value:.2f}")

#d


#problem 3

#a
insample_data = pd.read_csv('problem3_insample.csv')
outsample_data = pd.read_csv('problem3_outsample.csv')
insample_data['Date'] = pd.to_datetime(insample_data['Date'])
outsample_data['Date'] = pd.to_datetime(outsample_data['Date'])

insample_returns = insample_data.iloc[:, 1:]
outsample_returns = outsample_data.iloc[:, 1:]
asset_names = list(insample_returns.columns)

rf_annual = 0.04
rf_monthly = rf_annual / 12

expected_returns_monthly = insample_returns.mean()
ew_cov_monthly, ew_corr_monthly = ew_cov_corr_normalized(insample_returns, lam=0.97)

expected_returns_annual = (1 + expected_returns_monthly)**12 - 1
ew_cov_annual = ew_cov_monthly * 12

max_sr_weights, max_sr_ratio = max_sharpe_ratio(
    mu=expected_returns_annual.values,
    cov=ew_cov_annual.values,
    rf=rf_annual
)


print("assets weights:")
for i, asset in enumerate(asset_names):
    print(f"  {asset}: {max_sr_weights[i]:.4f} ({max_sr_weights[i]*100:.1f}%)")
print(f"SR: {max_sr_ratio:.4f}")

port_return_sr = max_sr_weights @ expected_returns_annual.values
port_vol_sr = np.sqrt(max_sr_weights @ ew_cov_annual.values @ max_sr_weights)

print(f"portfolio returns {port_return_sr:.4f} ({port_return_sr*100:.2f}%)")
print(f"portfolio Annual vol: {port_vol_sr:.4f} ({port_vol_sr*100:.2f}%)")

rp_weights = risk_parity_csd(ew_cov_annual.values)
    
  
for i, asset in enumerate(asset_names):
    print(f"  {asset}: {rp_weights[i]:.4f} ({rp_weights[i]*100:.1f}%)")

port_return_rp = rp_weights @ expected_returns_annual.values
port_vol_rp = np.sqrt(rp_weights @ ew_cov_annual.values @ rp_weights)
port_sharpe_rp = (port_return_rp - rf_annual) / port_vol_rp

print(f"portfolio returns: {port_return_rp:.4f} ({port_return_rp*100:.2f}%)")
print(f"portfolio Annual: {port_vol_rp:.4f} ({port_vol_rp*100:.2f}%)")
print(f"SR: {port_sharpe_rp:.4f}")

marginal_contrib = ew_cov_annual.values @ rp_weights / port_vol_rp
component_contrib = rp_weights * marginal_contrib

#b

portfolios = {
    'Max_Sharpe': max_sr_weights,
    'Risk_Parity': rp_weights
}

for port_name, weights in portfolios.items():
    attribution_result = asset_attribution(
        returns=outsample_returns,
        init_weights=weights
    )
    
    print("attribution result")
    print(attribution_result)
    print()
    
    total_returns = attribution_result.iloc[0, 1:-1].values
    return_attributions = attribution_result.iloc[1, 1:-1].values
    vol_attributions = attribution_result.iloc[2, 1:-1].values
    
    portfolio_return = attribution_result.iloc[0, -1]
    portfolio_vol_monthly = attribution_result.iloc[2, -1]
    portfolio_vol_annual = portfolio_vol_monthly * np.sqrt(12)
    

    print("assets | ex-ante weights | ex-post returns contributions| ex-post vol contributions")
    print("-" * 55)
    
    for i, asset in enumerate(asset_names):
        ex_ante_weight = weights[i]
        ex_post_return_contrib = return_attributions[i]
        ex_post_vol_contrib = vol_attributions[i]
        
        print(f"{asset:4s} | {ex_ante_weight:9.3f} | {ex_post_return_contrib:13.4f} | {ex_post_vol_contrib:13.4f}")
    
    print(f"  expost-portfolio_return: {portfolio_return:.4f}")
    print(f"  expost-portfolio_vol: {portfolio_vol_annual:.4f}")
    
    if port_name == 'Max_Sharpe':
        print(f"  ex-ante returns: {port_return_sr:.4f}")
        print(f"  ex-ante vol: {port_vol_sr:.4f}")
    else:
        print(f"  ex-ante returns: {port_return_rp:.4f}")
        print(f"  ex-ante vol: {port_vol_rp:.4f}")