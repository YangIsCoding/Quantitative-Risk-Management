import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy import optimize
from scipy.stats import norm
from scipy.integrate import quad
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import math
import itertools
import statsmodels.api as sm
import bisect
import time

# 在資料含有 NaN（缺值）時，計算共變異數矩陣（或相關係數矩陣）
def missing_cov(data, skipMiss=True, fun=np.cov):
    """
    Calculate covariance matrix handling missing values
    """
    if skipMiss:
        # Remove rows with any missing values
        valid_data = data[~np.isnan(data).any(axis=1)]
        if fun == np.cov:
            return np.cov(valid_data.T)
        elif fun == np.corrcoef:
            return np.corrcoef(valid_data.T)
        else:
            return fun(valid_data.T)
    else:
        # Pairwise calculation
        n_vars = data.shape[1]
        result = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                # Find valid pairs
                valid_mask = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
                valid_pairs = data[valid_mask][:, [i, j]]
                
                if len(valid_pairs) > 1:
                    if fun == np.cov:
                        result[i, j] = np.cov(valid_pairs[:, 0], valid_pairs[:, 1])[0, 1]
                    elif fun == np.corrcoef:
                        result[i, j] = np.corrcoef(valid_pairs[:, 0], valid_pairs[:, 1])[0, 1]
                    else:
                        temp_cov = fun(valid_pairs.T)
                        result[i, j] = temp_cov[0, 1] if temp_cov.ndim > 1 else temp_cov
                else:
                    result[i, j] = np.nan
        
        return result

# 計算 指數加權共變異數矩陣, 分析師會說：「越新的資料比較重要，越久遠的資料就該被淡忘一點。給風險管理、投資組合、VaR 模型用
def ewCovar(data, lambda_val):
    """
    Exponentially Weighted Covariance Matrix
    """
    n_obs, n_vars = data.shape
    weights = np.array([(1 - lambda_val) * (lambda_val ** i) for i in range(n_obs-1, -1, -1)])
    weights = weights / weights.sum()
    
    # Calculate weighted means
    weighted_mean = np.average(data, weights=weights, axis=0)
    
    # Center the data
    centered_data = data - weighted_mean
    
    # Calculate weighted covariance
    cov_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            cov_matrix[i, j] = np.average(centered_data[:, i] * centered_data[:, j], weights=weights)
    
    return cov_matrix

# 這支 near_psd 是在把「壞掉的」共變異數矩陣修好，讓它變成半正定（PSD），而且保留原本的變異數（對角線）。為什麼要修？因為用 pairwise、噪音、四捨五入或估計器不穩時，矩陣可能不是 PSD，接下來做馬可維茲最適化、Cholesky 分解、模擬就會直接爆掉。
def near_psd(matrix, epsilon=0.0):
    """
    Find the nearest positive semi-definite matrix preserving original variances
    Using Rebonato-Jäckel algorithm with proper scaling
    """
    # Ensure the matrix is symmetric
    A = (matrix + matrix.T) / 2
    
    # Extract standard deviations (preserve original variances)
    std_devs = np.sqrt(np.diag(A))
    
    # Convert to correlation matrix
    corr_matrix = A / np.outer(std_devs, std_devs)
    
    # Eigenvalue decomposition on correlation matrix
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    
    # Set negative eigenvalues to epsilon (typically 0)
    eigenvals_pos = np.maximum(eigenvals, epsilon)
    
    # Rebonato-Jäckel scaling step
    # scaling t_i = 1 / sum_j S_{ij}^2 * lam'_j
    eigenvecs_sq = eigenvecs**2
    denom = eigenvecs_sq @ eigenvals_pos
    denom = np.where(denom <= 0, 1.0, denom)
    t = 1.0 / denom
    
    # B = sqrt(T) S sqrt(Lam')
    B = (np.sqrt(t)[:, None]) * eigenvecs * (np.sqrt(eigenvals_pos)[None, :])
    corr_result = (B @ B.T + B @ B.T) / 2  # Ensure symmetry
    
    # Normalize to ensure diagonal is exactly 1
    d = np.sqrt(np.diag(corr_result))
    d = np.where(d <= 0, 1.0, d)
    corr_result = corr_result / np.outer(d, d)
    np.fill_diagonal(corr_result, 1.0)
    
    # Convert back to covariance matrix (restore original variances)
    result = corr_result * np.outer(std_devs, std_devs)
    
    return result

# Higham(2002) 最近正半定矩陣
    """什麼時候用 Higham、什麼時候用 RJ（near_psd 那支）？

Higham：

需要嚴格滿足「相關矩陣」的幾何約束（對角=1）並且找“最近”的 PSD。

收斂穩健、結果常更「正統」，但要迭代，可能比 RJ 慢一點。

RJ（eigen clipping + scaling）：

一次成形、速度快。

很適合快速把矩陣「修成能用」。

在某些幾何意義下不一定是「最近的」。
    """
def higham_nearestPSD(A, maxIts=100, tol=1e-8):
    """
    Higham's 2002 algorithm for finding the nearest PSD matrix preserving original variances
    """
    n = A.shape[0]
    
    # Ensure matrix is symmetric
    A = (A + A.T) / 2
    
    # Extract standard deviations (preserve original variances)
    std_devs = np.sqrt(np.diag(A))
    
    # Convert to correlation matrix
    corr_matrix = A / np.outer(std_devs, std_devs)
    
    # Initialize for Higham's algorithm on correlation matrix
    Y = corr_matrix.copy()
    Delta_S = np.zeros_like(corr_matrix)
    
    for k in range(maxIts):
        # Step 1: Project onto S (symmetric matrices with unit diagonal)
        # For correlation matrices, ensure diagonal is 1
        
        # Step 2: Project onto positive semidefinite cone
        eigenvals, eigenvecs = np.linalg.eigh(Y - Delta_S)
        eigenvals_pos = np.maximum(eigenvals, 1e-12)  # Use small positive threshold
        X = eigenvecs @ np.diag(eigenvals_pos) @ eigenvecs.T
        
        # Step 3: Update Delta_S
        Delta_S = X - (Y - Delta_S)
        
        # Step 4: Project back to correlation constraint (unit diagonal)
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)
        
        # Check convergence
        if np.linalg.norm(Y - Y_new, 'fro') <= tol:
            break
        
        Y = Y_new
    
    # Convert back to covariance matrix (restore original variances)
    result = Y * np.outer(std_devs, std_devs)
    
    # Final cleanup: ensure result is truly PSD (numerical precision fix)
    eigenvals_final, eigenvecs_final = np.linalg.eigh(result)
    eigenvals_final = np.maximum(eigenvals_final, 1e-12)
    result = eigenvecs_final @ np.diag(eigenvals_final) @ eigenvecs_final.T
    
    # Restore exact variances after final cleanup
    current_vars = np.diag(result)
    scale_factors = std_devs**2 / current_vars
    scale_matrix = np.sqrt(np.outer(scale_factors, scale_factors))
    result = result * scale_matrix
    
    return result

#幫共變異數矩陣安全地做 Cholesky 分解，不會因為數學小錯誤而報錯
"""想像你有一棟積木塔（這棟塔就是你的共變異數矩陣 🧱）。
正常情況下，這棟塔應該「底很穩」，這樣才能往上堆。
但有時候塔的底稍微歪一點（代表矩陣不是完全正定），
如果你直接堆積木（做 Cholesky 分解），塔就會倒掉（程式報錯 🚨）。

這個函式做的事是：

先試著直接堆積木（Cholesky 分解）
如果成功，就直接回傳結果。

如果塔倒了（報錯）

先幫塔「修一下底座」，讓它變穩（呼叫 near_psd() 把矩陣修成正半定）。

再用數學的方式保證塔每個基石都是正的（把負的特徵值變成 0）。

修好之後再堆一次積木，這次一定不會倒！

最後你就得到一個「穩定的下三角矩陣」，
可以拿來模擬金融風險、產生隨機報酬率、做 Monte Carlo 模擬等。"""

# Cholesky: A=L×LT
def chol_psd_simple(matrix):
    """
    Simple Cholesky decomposition that handles PSD matrices
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        # If not PD, find nearest PSD and decompose
        psd_matrix = near_psd(matrix)
        # Ensure truly PSD by setting small negative eigenvalues to zero
        eigenvals, eigenvecs = np.linalg.eigh(psd_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        psd_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        return np.linalg.cholesky(psd_matrix)

# 把資產價格（Price）轉換成報酬率（Return）
def return_calculate(prices, method="DISCRETE", dateColumn=None):
    """
    Calculate returns from prices
    """
    if isinstance(prices, pd.DataFrame):
        if dateColumn:
            # Keep date column but don't calculate returns for it
            price_cols = [col for col in prices.columns if col != dateColumn]
            
            # Create result DataFrame with proper column order
            result_data = {}
            
            if dateColumn in prices.columns:
                result_data[dateColumn] = prices[dateColumn].iloc[1:].reset_index(drop=True)
            
            # Calculate returns for all price columns at once
            if method.upper() == "DISCRETE":
                returns = prices[price_cols].pct_change().iloc[1:].reset_index(drop=True)
            elif method.upper() == "LOG":
                returns = np.log(prices[price_cols] / prices[price_cols].shift(1)).iloc[1:].reset_index(drop=True)
            
            # Add returns to result data
            for col in price_cols:
                result_data[col] = returns[col]
            
            # Create DataFrame with correct column order
            if dateColumn:
                columns = [dateColumn] + price_cols
            else:
                columns = price_cols
                
            result = pd.DataFrame(result_data, columns=columns)
            return result
        else:
            if method.upper() == "DISCRETE":
                return prices.pct_change().iloc[1:]
            elif method.upper() == "LOG":
                return np.log(prices / prices.shift(1)).iloc[1:]
    else:
        if method.upper() == "DISCRETE":
            return np.diff(prices) / prices[:-1]
        elif method.upper() == "LOG":
            return np.diff(np.log(prices))

# 用一樣多的錢，賺最多報酬、承擔最少風險, Sharpe Ratio 最大化
def super_efficient_portfolio(expected_rts,cov,rf=0.0425):
    """Given a target return, use assets to find the optimal portfolio with lowest risk"""
    fun=lambda wts: -(wts@expected_rts-rf)/np.sqrt(wts@cov@wts)
    x0 = np.full(expected_rts.shape[0],1/expected_rts.shape[0])
    cons = [{'type':'ineq', 'fun':lambda x:x},
        {'type':'eq', 'fun':lambda x:sum(x)-1}]
    bounds = [(0, 1) for _ in range(expected_rts.shape[0])]
    res = optimize.minimize(fun, x0, method='SLSQP',bounds=bounds,constraints=cons)
    return res

# 風險平價投資組合

"""🌈 小朋友版故事

想像你有三個朋友要一起搬一個大箱子 📦
（這個箱子就是「投資整體的風險」）

每個人力氣不同：

小熊 🐻 力氣大（波動高）

小兔 🐰 力氣中等

小狗 🐶 力氣小（波動低）

如果你讓小熊搬太多，他會累死；
小狗只搬一點又太閒。

所以最公平的辦法是：

「讓每個人都出一樣多的力氣。」💪

這樣大家一起搬，不會有人太重、有人太輕。
這就叫——風險平價（Risk Parity）。"""
def RiskParity(cov):
    """Given a target return, use assets to find the optimal portfolio with lowest risk"""
    fun=lambda w: (w*(cov@w)/np.sqrt(w@cov@w)).std()
    x0 = np.full(cov.shape[0],1/cov.shape[0])
    cons = [{'type':'ineq', 'fun':lambda x:x},
        {'type':'eq', 'fun':lambda x:sum(x)-1}]
    bounds = [(0, 1) for _ in range(cov.shape[0])]
    res = optimize.minimize(fun, x0, method='SLSQP',bounds=bounds,constraints=cons)
    return res

# 告訴你目前每個資產扛的風險比例是多少
def riskBudget(w,cov):
    """Calculate the portion of risk each stock of portfolio has. The sum of result is 1"""
    portfolioStd=np.sqrt(w@cov@w)
    Csd=w*(cov@w)/portfolioStd
    return Csd/portfolioStd

# gbsm 就是用一把「機率尺」去量未來可能賺到多少，
# 把它換算回今天的價值，告訴你這張選擇權票現在值多少。
def gbsm(s,strike,ttm,vol,rf,c,call=True):
    """
    Generalize Black Scholes Merton
    rf = c       -- Black Scholes 1973
    c = rf - q   -- Merton 1973 stock model where q is the continous dividend yield
    c = 0        -- Black 1976 futures option model
    c,r = 0      -- Asay 1982 margined futures option model
    c = rf - rff -- Garman and Kohlhagen 1983 currency option model where rff is the risk free rate of the foreign currency

    Option valuation via BSM closed formula
    European Style.  Assumed LogNormal Prices
    s - Underlying Price
    strike - Strike Price
    ttm - time to maturity
    rf - Risk free rate
    vol - Yearly Volatility
    c - Cost of Carry
    call - Call valuation if set True
    """
    d1=(np.log(s/strike)+(c+vol**2/2)*ttm)/vol/np.sqrt(ttm)
    d2=d1-vol*np.sqrt(ttm)
    if call:
        return s*np.exp((c-rf)*ttm)*norm.cdf(d1)-strike*np.exp(-rf*ttm)*norm.cdf(d2)
    else:
        return strike*np.exp(-rf*ttm)*norm.cdf(-d2)-s*np.exp((c-rf)*ttm)*norm.cdf(-d1)

# 這個函式不是在算「票值多少」，
# 而是在算「票價對各種因素的靈敏度」。
# 6 個旋鈕（Delta/Gamma/Vega/Theta/Rho/Carry Rho）讓你知道：
# 價格、時間、波動、利率、持有成本各動一點點，票價會跟著怎麼動。
def greeks_closed_form(s,strike,ttm,vol,rf,c,call=True):
    """Closed from for greeks calculation from Generalize Black Scholes Merton
        Generalize Black Scholes Merton:
        rf = c       -- Black Scholes 1973
        c = rf - q   -- Merton 1973 stock model where q is the continous dividend yield
        c = 0        -- Black 1976 futures option model
        c,r = 0      -- Asay 1982 margined futures option model
        c = rf - rff -- Garman and Kohlhagen 1983 currency option model where rff is the risk free rate of the foreign currency

        Option valuation via BSM closed formula
        European Style.  Assumed LogNormal Prices
        s - Underlying Price
        strike - Strike Price
        ttm - time to maturity
        rf - Risk free rate
        vol - Yearly Volatility
        c - Cost of Carry
        call - Call valuation if set True
    """
    d1=(np.log(s/strike)+(c+vol**2/2)*ttm)/vol/np.sqrt(ttm)
    d2=d1-vol*np.sqrt(ttm)
    optionType=['Call'] if call else ['Put']
    ans=pd.DataFrame(index=optionType,columns=['Detla','Gamma','Vega','Theta','Rho','Carry Rho'])
    if call:
        ans['Detla'] = np.exp((c-rf)*ttm)*norm.cdf(d1,loc=0,scale=1)
        ans['Theta'] = -s*np.exp((c-rf)*ttm)*norm.pdf(d1,loc=0,scale=1)*vol/(2*np.sqrt(ttm))-(c-rf)*s*np.exp((c-rf)*ttm)*norm.cdf(d1,loc=0,scale=1)-rf*strike*np.exp(-rf*ttm)*norm.cdf(d2,loc=0,scale=1)
        ans['Rho'] = ttm*strike*np.exp(-rf*ttm)*norm.cdf(d2,loc=0,scale=1)
        ans['Carry Rho'] = ttm*s*np.exp((c-rf)*ttm)*norm.cdf(d1,loc=0,scale=1)
    else:
        ans['Detla'] = np.exp((c-rf)*ttm)*(norm.cdf(d1,loc=0,scale=1)-1)
        ans['Theta'] = -s*np.exp((c-rf)*ttm)*norm.pdf(d1,loc=0,scale=1)*vol/(2*np.sqrt(ttm))+(c-rf)*s*np.exp((c-rf)*ttm)*norm.cdf(-d1,loc=0,scale=1)+rf*strike*np.exp(-rf*ttm)*norm.cdf(-d2,loc=0,scale=1)
        ans['Rho'] = -ttm*strike*np.exp(-rf*ttm)*norm.cdf(-d2,loc=0,scale=1)
        ans['Carry Rho'] = -ttm*s*np.exp((c-rf)*ttm)*norm.cdf(-d1,loc=0,scale=1)
    ans['Gamma'] = norm.pdf(d1,loc=0,scale=1)*np.exp((c-rf)*ttm)/(s*vol*np.sqrt(ttm))
    ans['Vega'] = s*np.exp((c-rf)*ttm)*norm.pdf(d1,loc=0,scale=1)*np.sqrt(ttm)

    return ans

"""用最大概似法 (MLE) 配適分佈.
你有很多糖果（資料 data），
你想知道：「這些糖果的大小是不是符合某一種形狀？」
例如：

是像「正態分佈」（中間多、兩邊少）嗎？

還是「t 分佈」（尾巴比較厚）？

那這個 FittedModel 類別就是一個「科學實驗機器」，
幫你用數學方式配出最接近資料的形狀！"""
class FittedModel:
    """The prototype of fitted distribution."""
    def __init__(self):
        self.dist=self.set_dist()
        self.frz_dist=None

    def set_dist(self):
        """Need to be implemented in subclass to set the dist."""
        raise NotImplementedError
    
    def freeze_dist(self,parameters):
        """Need to be implemented in subclass to set the parameters of different distribution."""
        raise NotImplementedError

    def fit(self,data,x0,cons):
        """
        Use MLE to fit the distribution
        x0 is initial paremeters which needed to be implemented in subclass
        cons is constraints of parameters which needed to be implemented in subclass
        """
        def nll(parameters,x):
            """Negative likelihood function"""
            self.freeze_dist(parameters)
            ll=self.frz_dist.logpdf(x=x).sum()
            return -ll
        MLE = minimize(nll, x0=x0, args=data, constraints = cons)
        self.freeze_dist(MLE.x)
        self.fitted_parameters=MLE.x

    @property
    def fitted_dist(self):
        return self.frz_dist

"""想像你有一台「配糖果形狀的機器」 (FittedModel)，
它很聰明，但還不知道自己要配哪種糖果。
所以你幫它生了一個小孩，叫做 Norm。

這個小孩專門研究：

「大部分糖果中等大小，少部分特別大或特別小」
這種形狀的糖果，就是 正態分佈（Normal Distribution） 🍬。"""
class Norm(FittedModel):
    def set_dist(self):
        """set the distribution to be normal"""
        return stats.norm
        
    def freeze_dist(self,parameters):
        """set the parameters of norm: parameters[0]--mu, parameters[1]--std"""
        self.frz_dist=self.dist(loc=parameters[0],scale=parameters[1])

    def fit(self,data):
        """set the initial parameters and cons to call the father's fit"""
        x0 = (data.mean(),data.std())
        cons = [ {'type':'ineq', 'fun':lambda x:x[1]} ]
        super().fit(data,x0,cons)
"""還記得 Norm 是研究「正常糖果」的機器人嗎？🍬
那 T 就是牠的「兄弟」🤖，
不過這個兄弟比較「頑皮」一點，因為他研究的是：

「有時會出現很極端糖果的世界！」🍭🍭🍭
這種情況就像資料裡有很多「離群值」（outliers），
所以用 t 分佈 會比 正態分佈 更準。"""
class T(FittedModel):
    def set_dist(self):
        """set the distribution to be normal"""
        return stats.t
        
    def freeze_dist(self,parameters):
        """set the parameters of norm: parameters[0]--degree of freedom, parameters[1]--mu, parameters[2]--std"""
        self.frz_dist=self.dist(df=parameters[0],loc=parameters[1],scale=parameters[2])
        
    def fit(self,data):
        """set the initial parameters and cons to call the father's fit"""
        cons=[ {'type':'ineq', 'fun':lambda x:x[0]-2} , {'type':'ineq', 'fun':lambda x:x[2]} ] 
        mu=data.mean()
        df=6/stats.kurtosis(data,bias=False)+4
        df = 2.5 if df<=2 else df
        std=np.sqrt(data.var()*df/(df-2))
        x0 = np.array([df,mu,std])
        super().fit(data,x0,cons)

"""還記得嗎？
t 分佈 是一種「尾巴比較厚」的形狀，
它用三個參數描述糖果的分佈：

參數	意思	比喻
df	尾巴厚度（自由度）	控制尾巴有多「肥」 🍩
mu	平均值	糖果堆的中心位置 🎯
std	標準差	糖果散得多寬（胖或瘦） 🍬

而這個 T_mean0 的想法是：

「有些糖果我知道中心一定在 0，不需要電腦去學。」"""
class T_mean0(FittedModel):
    def set_dist(self):
        """set the distribution to be normal"""
        return stats.t
        
    def freeze_dist(self,parameters):
        """set the parameters of norm: parameters[0]--degree of freedom, parameters[1]--mu, parameters[2]--std"""
        self.frz_dist=self.dist(df=parameters[0],loc=parameters[1],scale=parameters[2])
        
    def fit(self,data):
        """set the initial parameters and cons to call the father's fit"""
        cons=[ {'type':'ineq', 'fun':lambda x:x[0]-2} , {'type':'eq', 'fun':lambda x:x[1]},{'type':'ineq', 'fun':lambda x:x[2]} ] 
        mu=data.mean()
        df=6/stats.kurtosis(data,bias=False)+4
        df = 2.5 if df<=2 else df
        std=np.sqrt(data.var()*df/(df-2))
        x0 = np.array([df,mu,std])
        super().fit(data,x0,cons)

"""還記得你前面做的那些分佈機器人嗎？
像是：

機器人	功能
Norm	學習「鐘形曲線」的世界（正態分佈）📈
T	學習「尾巴比較厚」的世界（t 分佈）🍩
T_mean0	尾巴厚、但中心固定在 0 的世界 🎯

這些都是單一模型，
他們可以幫你學「一組資料」的形狀。

可是如果你有一整個表格要學呢？
像是：

Stock A	Stock B	Stock C
0.01	-0.02	0.03
0.05	0.01	-0.02
...	...	...

你就不想一個一個手動 fit 😵。
這時候你就派出：

🚀 ModelFitter 機器人總控中心！"""
class ModelFitter:
    """ Fit the data with Model, return a group of fitted distributions

        Parameters:
            FittedModel(Class) ---- a subclass of FittedModel class

        Usage:
            dists=ModelFitter(FittedModel).fit(data)
    """

    def __init__(self,FittedModel):
        """ Initialize the model within the class to fit all the data."""
        self.model=FittedModel()
    
    def fit(self,data):
        """Fit all the data with the model inside the Fitter
            Data(Dataframe) -- return of stock
        """
        dists=[]
        for name in data.columns:
            rt=np.array(data[name].values)
            self.model.fit(rt)
            dists.append(self.model.fitted_dist)
        dists=pd.DataFrame(dists).T
        dists.columns=data.columns
        dists.index=["distribution"]
        return dists

# NotPsdError 是一個自訂的「警報器」，
# 當輸入的矩陣不是「穩定又對稱」的正定矩陣時，
# 它會大喊：「停下來！這樣會倒！」
class NotPsdError(Exception):
    """ 
    Used for expection raise if the input matrix is not sysmetric positive definite 
    """
    pass

"""蓋積木塔的檢查員

你有一個超大積木塔（這個塔就是一個「矩陣」）。
想要把它拆成「一層一層的積木」堆回去（這就叫 Cholesky 分解）：

把塔 A 變成「下三角積木 L」再乘上它的翻面 
A = L × Lᵀ

這個 chol_psd_class 就是檢查員 + 工人：

先檢查塔的每一層是不是夠穩（要「正定/半正定」才行）

再按順序把每一層拆成積木，做出 L

如果遇到不穩的地方（有一層是負的），就大喊：

Not PSD!（代表塔不合格，會倒！）"""
class chol_psd_class():
    """
    Cholesky Decompstion: Sysmetric Positive Definite matrix could use Cholesky 
    algorithm to fatorize the matrix to the product between a lower triangle matrix and
    upper triangle matrix

    Parameter:
        matrix(Array)  --  Sysmetric Positive Definite (or Positive Semi-definite) 
                    matrix needed to do Cholesky Factorization.
    
    Formula: 
        matrix=L*L.T

    Usage:
        Chol_model=chol_psd_class(matrix)
        root=Chol_model.root
    """
    def __init__(self,matrix):
        self.__psd=matrix
        self.run()

    def run(self):
        n=self.__psd.shape[0]
        root=np.zeros([n,n])
        for i in range(n):
            root[i][i] = self.__psd[i][i] - root[i][:i] @ root[i][:i].T
            root[i][i]=0 if 0>=root[i][i]>=-1e-8 else root[i][i]
            if root[i][i]<0:
                raise NotPsdError("Not PSD!")
            root[i][i]=np.sqrt(root[i][i])
            
            if root[i][i]==0:
                continue
            for j in range(i+1,n):
                root[j][i]=(self.__psd[j][i]-root[i,:i] @ root[j,:i])/root[i][i]
        self.__root=root
        self.__ispsd=True

    @property
    def root(self):
        return self.__root   
    
    @property
    def ispsd(self):
        return self.__ispsd 

# Weighted_F_norm 是一個「比較矩陣像不像」的工具，
#它會考慮哪些部分比較重要，
#所以叫做「加權版的距離尺」。
#這種量測方式叫：
#加權 Frobenius 範數（Weighted Frobenius Norm）
class Weighted_F_norm:
    """
    Given the weight matrix(Array), calculate the Weighted Frobenius Norm. (Assume it's diagonal)
    """
    def compare_F(self,mat_a,mat_b,mat_w):
        """Give two matrix, use Weighted Frobenius Norm to calculate how close they are"""
        err = mat_a-mat_b
        weighted_err = np.sqrt(mat_w) @ err @ np.sqrt(mat_w) 
        w_F_norm = np.sqrt(np.square(weighted_err).sum())
        return w_F_norm
    
    def calculate_F(self,mat,mat_w):
        "Given one matrix, calculate its Weighted Frobenius Norm"
        weighted_err = np.sqrt(mat_w) @ mat @ np.sqrt(mat_w)
        w_F_norm = np.sqrt(np.square(weighted_err).sum())
        return w_F_norm

class NotSysmetricError(Exception):
    """ 
    Used for expection raise if the input matrix is not sysmetric
    """
    pass

# 特徵值 ≥ 0，沒有塌陷
class NegativeEigError(Exception):
    """ 
    Used for expection raise if matrix has the negative eigvalue
    """
    pass

# PSD.confirm() 就像是「矩陣安全檢查員」，
# 它會確保這個矩陣的形狀對稱、底座正、能安全拆開。
# 如果任何一關不合格，就舉牌大喊：
# 🚨「這個矩陣不穩！」
class PSD:
    """
    PSD class is used for Positive Semi-Definite Matrix Confirmation.
    psd(Array) -- matrix to be confirmed
    """
    @staticmethod
    def confirm(psd):
        if not np.allclose(psd,psd.T):
            raise NotSysmetricError("Matrix does not equal to Matrix.T")
        eig_val=np.linalg.eigvals(psd)
        neg_eig=len(eig_val[eig_val<0])
        if neg_eig==0 or chol_psd_class(psd).ispsd:
            print("Matrix is Sysmetric Positive Definite!")
            return True
        else:
            raise NegativeEigError("Matrix has negative eigenvalue.")

# near_psd_class = 用 RJ 方法把「壞掉的共變異數矩陣」修成可用且穩的版本，
# 並用「加權尺」告訴你修了多少。
# 金融上非常實用：不管是最適化、模擬、分解都能更安心地跑起來。
class near_psd_class(Weighted_F_norm):
    """
    Rebonato and Jackel's Method to get acceptable PSD matrix 
    
    Parameters:
        not_psd (Array) -- the matrix which is not positive semi-definite matrix
        weight (Array) -- used for calculating the Weighted Frobenius Norm (Assume it's diagonal)

    Usage:
        near_psd_model=near_psd_class(non_psd,weight)
        psd=near_psd_model.psd
    """
    def __init__(self,not_psd,weight):
        self.__not_psd=not_psd
        self.__weight=weight
        self.run()
        self.F_compare_norm(weight)
        
    def run(self):
        n=self.__not_psd.shape[0]
        invSD = np.eye(n)
        corr=self.__not_psd
        if not np.allclose(np.diag(self.__not_psd),np.ones(n)):
            invSD=np.diag(1/np.sqrt(np.diag(self.__not_psd)))
            corr=invSD @ self.__not_psd @ invSD
        eig_val,eig_vec=np.linalg.eigh(corr)
        eig_val[eig_val<0]=0
        scale_mat = np.diag(1/(eig_vec * eig_vec @ eig_val))
        B = np.sqrt(scale_mat) @ eig_vec @ np.sqrt(np.diag(eig_val))
        corr=B @ B.T
        SD=np.diag(1/np.diag(invSD))
        psd = SD @ corr @ SD
        self.__psd = psd

    def F_compare_norm(self,weight):
        self.__F = self.compare_F(self.__psd,self.__not_psd,weight)

    @property
    def psd(self):
        return self.__psd
    
    @property
    def F(self):
        return self.__F
    
    
# 你每天記錄很多股票的「心情起伏」（報酬）。
"""但我們覺得：越新的日子越重要、很久以前就沒那麼重要。
EWMA 就是一台把「新的分數權重比較大」的機器，來算：

大家一起晃動的程度：共變異數矩陣（cov）

只看「一起、不同步」的程度：相關矩陣（corr）"""
class EWMA:
    """
    Calculate the Exponentially Weighted Covariance & Correaltion Matrix
    
    Parameter: 
        data (Array)  -- return data for calculating Covariance & Correaltion Matrix (array)
        lambda_(Float)  -- smoothing parameter (less than 1)
        flag (Boolean) -- a flag (optional) to dertermine whether to subtract mean from data.
                            if it set False, data would not subtract its mean.

    fomula: \\sigma_t^2=\\lambda \\sigma_{t-1}^2+(1-\\lambda)r_{t-1}^2

    Usage:  
        model=EWMA(data,0.97)
        cov_mat=model.cov_mat
        corr_mat=model.corr_mat
    """
    def __init__(self,data,lambda_,flag=False):
        self.__data=data if flag==False else data-data.mean(axis=0)
        self.__lambda=lambda_
        self.get_weight() 
        self.cov_matrix()
        self.corr_matrix()

    def get_weight(self):
        n=self.__data.shape[0]
        weight_mat=[(1-self.__lambda)*self.__lambda**(n-i-1) for i in range(n)]
        self.__weight_mat=np.diag(weight_mat)

    def cov_matrix(self):
        self.__cov_mat=self.__data.T @ self.__weight_mat @ self.__data

    def corr_matrix(self):
        n=self.__data.shape[1]
        invSD=np.sqrt(1./np.diag(self.__cov_mat))
        invSD=np.diag(invSD)
        self.__corr_mat=invSD @ self.__cov_mat @ invSD
        return self.__corr_mat

    def plot_weight(self,k=None,ax=None,label=None):
        weight=np.diag(self.__weight_mat)[::-1]
        cum_weight=weight.cumsum()/weight.sum()
        sns.lineplot(cum_weight,ax=ax,label="{:.2f}".format(label) if label!=None else "")
        if ax!=None:
            ax.set_xlabel('Time Lags')
            ax.set_ylabel('Cumulative Weights')
            ax.set_title("Weights of differnent lambda")
        ax.legend(loc='best')

    @property
    def cov_mat(self):
        return self.__cov_mat    

    @property
    def corr_mat(self):
        return self.__corr_mat

"""共變異數矩陣（cov）：每個資產「晃多大」＋「彼此一起晃多少」。

相關矩陣（corr）：只看「同步程度」，不管誰晃得大或小（都先變成同等級）。

EW（指數加權）：最近發生的事比較重要，舊的事慢慢淡掉。

這台機器做四種「醬汁」：

EW_cov_corr()
用「最近比較重要」算出每個資產自己的晃動大小（EW 共變異數的標準差），
再配上一般的「同步程度」（平常的 corr）
→ 變出一罐「大小用 EW、同步用一般」的醬。

EW_corr_cov()
反過來：「同步程度」用 EW 的（最近更重要），
「大小」用一般的 cov 的標準差
→ 「大小用一般、同步用 EW」。

EW_corr_EW_cov()
大小與同步都用 EW（兩個都「最近比較重要」）。

corr_cov()
都用一般的（大小用一般 cov 的標準差；同步用一般 corr）。

把「大小」記成一個對角矩陣 
D（對角線放每個資產的標準差 σ），
「同步」記成 
C（相關矩陣），
就用公式：
Σ=DCD

把它們重新「合體」成一個共變異數矩陣。"""
class Cov_Generator:
    """
    Convariance Derivation through differnet combination of EW covariance, EW correlation, covariance and correlation.

    Parameter:
        data(Array) -- data which is used for get the EW covariance, EW correlation, covariance and correlation
    
    Usage:
        cov_generator=Cov_Generator(data)
        cov_generator.EW_cov_corr()
        cov_generator.EW_corr_cov()
        cov_generator.EW_corr_EW_cov()
        cov_generator.corr_cov()
    """
    def __init__(self,data):
        self.__data = data
        self.__EWMA = EWMA(data,0.97)
        self.__EW_cov = self.__EWMA.cov_mat
        self.__EW_corr = self.__EWMA.corr_mat
        self.__cov = np.cov(data.T)
        invSD=np.diag(1/np.sqrt(np.diag(self.__cov)))
        self.__corr = invSD @ self.__cov @ invSD

    def EW_cov_corr(self):
        std=np.diag(np.diag(self.__EW_cov))
        return std @ self.__corr @ std

    def EW_corr_cov(self):
        std=np.diag(np.diag(self.__cov))
        return std @ self.__EW_corr @ std

    def EW_corr_EW_cov(self):
        std=np.diag(np.diag(self.__EW_cov))
        return std @ self.__EW_corr @ std

    def corr_cov(self):
        std=np.diag(np.diag(self.__cov))
        return std @ self.__corr @ std
"""你有一張「同學彼此多合拍」的大表（相關/共變異數矩陣）。
有時候這張表壞掉了，數學上不合理（會讓後面算東西倒掉）。
Higham_psd 就是一位修表的工程師，用「來回拉直」的方法把表修好。

它有兩個魔法動作：

Projection_U：把每個人自己對自己設成 1
就像提醒大家：「自己跟自己要滿分 1 喔！」
（相關矩陣的對角線一定是 1）

Projection_S：把整張表拉回「不會壞掉」的範圍
用一把「權重尺」量（Weighted Frobenius Norm），
找出壞掉的地方（負的特徵值），把它們調成不壞（>=0），
讓整張表變成正半定（安全、可用）。

工程師會一直交替做這兩件事：
先拉回不壞（S），再把對角變 1（U），
S → U → S → U …
直到表格幾乎不再改變，代表修好了 ✅

最後給你兩樣東西：

psd：修好、能用、不會倒的矩陣

F / F_norm：用那把「權重尺」量，修前修後差多少（越小表示改動越少）"""
class Higham_psd(Weighted_F_norm,chol_psd_class):
    """
    Higham's Method to get nearest PSD matrix under the measure of Weighted Frobenius Norm
    
    Parameters:
        not_psd (Array) -- the matrix which is not positive semi-definite matrix
        weight (Array) -- used for calculating the Weighted Frobenius Norm (Assume it's diagonal)
        epsilon (Float)-- the acceptable precision between near_psd and non_psd
        max_iter (Integer)-- maximum iteration number

    Usage:
        Higham_psd_model=Higham_psd(non_psd,weight)
        psd=Higham_psd_model.psd
    """
    def __init__(self,not_psd,weight,epsilon=1e-9,max_iter=1e10):
        self.__not_psd=not_psd
        self.__weight=weight
        self.run(epsilon=epsilon,max_iter=max_iter)
        self.F_compare_norm(weight)

    def Projection_U(self,A):
        A_copy=A.copy()
        np.fill_diagonal(A_copy,1)
        return A_copy
        
    def Projection_S(self,A):
        w_sqrt=np.sqrt(self.__weight)
        eig_val,eig_vec=np.linalg.eigh(w_sqrt @ A @ w_sqrt)
        eig_val[eig_val<0]=0
        A_plus=eig_vec @ np.diag(eig_val) @ eig_vec.T
        w_sqrt_inv=np.diag(1/np.diag(w_sqrt))
        ans = w_sqrt_inv @ A_plus @ w_sqrt_inv
        return ans
    
    def run(self,epsilon,max_iter):
        Y=self.__not_psd
        F1=np.inf
        F2=self.calculate_F(Y,self.__weight)
        delta=0
        iteration=0
        neg_eig=0
        while abs(F1-F2)>epsilon or neg_eig>0:
            R=Y-delta
            X=self.Projection_S(R)
            delta=X-R
            Y=self.Projection_U(X)
            F1,F2=F2,self.calculate_F(Y,self.__weight)
            iteration+=1
            if iteration>max_iter:
                break
            eig_val=np.linalg.eigvals(Y)
            neg_eig=len(eig_val[eig_val<0])

        self.__F_norm=F2
        self.__psd=Y

    def F_compare_norm(self,weight):
        self.__F = self.compare_F(self.__psd,self.__not_psd,weight)
        
    @property
    def psd(self):
        return self.__psd 
    @property
    def F_norm(self):
        return self.__F_norm 
    
    @property
    def F(self):
        return self.__F

"""想像你在觀察好多隻小狗每天的心情變化 🐶🐶🐶
你發現：越新的心情比較重要，舊的記錄就慢慢淡忘。

EWMA 就是這樣一台「最近的事情更重要」的記錄機器。

λ（lambda）越大 → 記性越好（慢慢忘）

λ越小 → 記性差，只看最近幾天

例如 λ=0.97 時，代表你會保留大約最近 1/(1−0.97)=33 天的影響力。"""
class EWMA:
    """
    Calculate the Exponentially Weighted Covariance & Correaltion Matrix
    
    Parameter: 
        data (Array)  -- return data for calculating Covariance & Correaltion Matrix (array)
        lambda_(Float)  -- smoothing parameter (less than 1)
        flag (Boolean) -- a flag (optional) to dertermine whether to subtract mean from data.
                            if it set False, data would not subtract its mean.

    fomula: \\sigma_t^2=\\lambda \\sigma_{t-1}^2+(1-\\lambda)r_{t-1}^2

    Usage:  
        model=EWMA(data,0.97)
        cov_mat=model.cov_mat
        corr_mat=model.corr_mat
    """
    def __init__(self,data,lambda_,flag=False):
        self.__data=data if flag==False else data-data.mean(axis=0)
        self.__lambda=lambda_
        self.get_weight() 
        self.cov_matrix()
        self.corr_matrix()

    def get_weight(self):
        n=self.__data.shape[0]
        weight_mat=[(1-self.__lambda)*self.__lambda**(n-i-1) for i in range(n)]
        self.__weight_mat=np.diag(weight_mat)

    def cov_matrix(self):
        self.__cov_mat=self.__data.T @ self.__weight_mat @ self.__data

    def corr_matrix(self):
        n=self.__data.shape[1]
        invSD=np.sqrt(1./np.diag(self.__cov_mat))
        invSD=np.diag(invSD)
        self.__corr_mat=invSD @ self.__cov_mat @ invSD
        return self.__corr_mat

    def plot_weight(self,k=None,ax=None,label=None):
        weight=np.diag(self.__weight_mat)[::-1]
        cum_weight=weight.cumsum()/weight.sum()
        sns.lineplot(cum_weight,ax=ax,label="{:.2f}".format(label) if label!=None else "")
        if ax!=None:
            ax.set_xlabel('Time Lags')
            ax.set_ylabel('Cumulative Weights')
            ax.set_title("Weights of differnent lambda")
        ax.legend(loc='best')

    @property
    def cov_mat(self):
        return self.__cov_mat    

    @property
    def corr_mat(self):
        return self.__corr_mat

# PCA模擬函數 - 主成分分析降維模擬
def pca_simulation(cov_matrix, n_samples=100000, explained_variance_ratio=0.99, random_seed=None):
    """
    使用主成分分析進行降維模擬
    
    Parameters:
    -----------
    cov_matrix : array_like
        協方差矩陣
    n_samples : int, default=100000
        模擬樣本數
    explained_variance_ratio : float, default=0.99
        累積方差解釋度閾值（保留主成分的標準）
    random_seed : int, optional
        隨機種子
        
    Returns:
    --------
    simulated_data : ndarray
        模擬的數據
    n_components : int
        保留的主成分數量
    eigenvals : ndarray
        特徵值（按大小排序）
    eigenvecs : ndarray
        特徵向量（按特徵值大小排序）
        
    Notes:
    ------
    PCA模擬步驟:
    1. 特徵值分解：Σ = QΛQ'
    2. 選擇成分：保留指定累積方差解釋度的主成分
    3. 降維模擬：在主成分空間生成隨機數
    4. 空間還原：轉換回原始變數空間
    
    數學公式:
    - Z ~ N(0,I_k) (k維獨立隨機數)
    - Y = Z√Λₖ (縮放到主成分方差)
    - X = YQₖ' (轉換回原始空間)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 步驟1: 特徵值分解
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # 步驟2: 按特徵值大小排序（從大到小）
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # 步驟3: 選擇解釋指定方差比例的主成分數量
    cumulative_explained = np.cumsum(eigenvals) / np.sum(eigenvals)
    n_components = np.argmax(cumulative_explained >= explained_variance_ratio) + 1
    
    # 步驟4: 保留選定的主成分
    selected_eigenvals = eigenvals[:n_components]
    selected_eigenvecs = eigenvecs[:, :n_components]
    
    # 步驟5: 在主成分空間生成隨機數
    Z = np.random.randn(n_samples, n_components)
    
    # 步驟6: 縮放到主成分方差
    scaled_Z = Z * np.sqrt(selected_eigenvals)
    
    # 步驟7: 轉換回原始變數空間
    simulated_data = scaled_Z @ selected_eigenvecs.T
    
    return simulated_data, n_components, eigenvals, eigenvecs

"""標準差 (大小) × 相關係數 (同步程度) × 標準差」來拼出各種共變異數矩陣。"""
class Cov_Generator:
    """
    Convariance Derivation through differnet combination of EW covariance, EW correlation, covariance and correlation.

    Parameter:
        data(Array) -- data which is used for get the EW covariance, EW correlation, covariance and correlation
    
    Usage:
        cov_generator=Cov_Generator(data)
        cov_generator.EW_cov_corr()
        cov_generator.EW_corr_cov()
        cov_generator.EW_corr_EW_cov()
        cov_generator.corr_cov()
    """
    def __init__(self,data):
        self.__data = data
        self.__EWMA = EWMA(data,0.97)
        self.__EW_cov = self.__EWMA.cov_mat
        self.__EW_corr = self.__EWMA.corr_mat
        self.__cov = np.cov(data.T)
        invSD=np.diag(1/np.sqrt(np.diag(self.__cov)))
        self.__corr = invSD @ self.__cov @ invSD

    def EW_cov_corr(self):
        std=np.diag(np.diag(self.__EW_cov))
        return std @ self.__corr @ std

    def EW_corr_cov(self):
        std=np.diag(np.diag(self.__cov))
        return std @ self.__EW_corr @ std

    def EW_corr_EW_cov(self):
        std=np.diag(np.diag(self.__EW_cov))
        return std @ self.__EW_corr @ std

    def corr_cov(self):
        std=np.diag(np.diag(self.__cov))
        return std @ self.__corr @ std
# ===== 第12章：選擇權定價與希臘字母計算 =====

def gbsm_with_greeks(option_type, underlying, strike, ttm, rf, c, vol):
    """
    使用GBSM模型計算歐式選擇權的價格和希臘字母
    
    Parameters:
    -----------
    option_type : str
        選擇權類型 ('Call' or 'Put')
    underlying : float
        標的資產價格
    strike : float  
        履約價格
    ttm : float
        到期時間（年）
    rf : float
        無風險利率
    c : float
        持有成本（carry rate = rf - dividend_rate）
    vol : float
        隱含波動率
        
    Returns:
    --------
    dict : 包含價格和所有希臘字母的字典
    """
    is_call = (option_type == 'Call')
    
    # 計算d1和d2
    d1 = (np.log(underlying/strike) + (c + vol**2/2)*ttm) / (vol*np.sqrt(ttm))
    d2 = d1 - vol*np.sqrt(ttm)
    
    # 使用existing gbsm函數計算選擇權價格
    price = gbsm(underlying, strike, ttm, vol, rf, c, call=is_call)
    
    # 計算希臘字母
    if is_call:
        delta = np.exp((c-rf)*ttm) * norm.cdf(d1)
        theta = (-underlying*np.exp((c-rf)*ttm)*norm.pdf(d1)*vol/(2*np.sqrt(ttm)) 
                - (c-rf)*underlying*np.exp((c-rf)*ttm)*norm.cdf(d1) 
                - rf*strike*np.exp(-rf*ttm)*norm.cdf(d2))
        rho = ttm*strike*np.exp(-rf*ttm)*norm.cdf(d2)
    else:
        delta = np.exp((c-rf)*ttm) * (norm.cdf(d1) - 1)
        theta = (-underlying*np.exp((c-rf)*ttm)*norm.pdf(d1)*vol/(2*np.sqrt(ttm)) 
                + (c-rf)*underlying*np.exp((c-rf)*ttm)*norm.cdf(-d1) 
                + rf*strike*np.exp(-rf*ttm)*norm.cdf(-d2))
        rho = -ttm*strike*np.exp(-rf*ttm)*norm.cdf(-d2)
    
    # Gamma對Call和Put都一樣
    gamma = norm.pdf(d1)*np.exp((c-rf)*ttm)/(underlying*vol*np.sqrt(ttm))
    
    # Vega對Call和Put都一樣
    vega = underlying*np.exp((c-rf)*ttm)*norm.pdf(d1)*np.sqrt(ttm)
    
    return {
        'Value': price,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    }

def binomial_american_option(is_call, S, K, T, r, c, vol, N=500):
    """
    使用二元樹方法計算美式選擇權價格
    
    Parameters:
    -----------
    is_call : bool
        True為買權，False為賣權
    S : float
        標的資產價格
    K : float
        履約價格
    T : float
        到期時間（年）
    r : float
        無風險利率
    c : float
        持有成本（carry rate）
    vol : float
        波動率
    N : int
        二元樹步數
        
    Returns:
    --------
    float : 選擇權價格
    """
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    
    # 風險中性機率
    p = (np.exp(c * dt) - d) / (u - d)
    
    # 初始化資產價格樹
    S_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            S_tree[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # 初始化選擇權價值樹
    option_tree = np.zeros((N + 1, N + 1))
    
    # 到期時的選擇權價值
    for j in range(N + 1):
        if is_call:
            option_tree[j, N] = max(0, S_tree[j, N] - K)
        else:
            option_tree[j, N] = max(0, K - S_tree[j, N])
    
    # 向後遞推
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # 歐式價值（折現期望值）
            european_value = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            
            # 提前執行價值
            if is_call:
                exercise_value = max(0, S_tree[j, i] - K)
            else:
                exercise_value = max(0, K - S_tree[j, i])
            
            # 美式選擇權價值為兩者最大值
            option_tree[j, i] = max(european_value, exercise_value)
    
    return option_tree[0, 0]

def calculate_american_greeks_numerical(is_call, S, K, T, r, c, vol, N=500):
    """
    使用數值方法計算美式選擇權的希臘字母
    
    Parameters:
    -----------
    is_call : bool
        True為買權，False為賣權
    S : float
        標的資產價格
    K : float
        履約價格
    T : float
        到期時間（年）
    r : float
        無風險利率
    c : float
        持有成本（carry rate）
    vol : float
        波動率
    N : int
        二元樹步數
        
    Returns:
    --------
    dict : 包含價格和所有希臘字母的字典
    """
    base_price = binomial_american_option(is_call, S, K, T, r, c, vol, N)
    
    # Delta (對標的資產價格的偏微分)
    dS = 0.01 * S
    price_up = binomial_american_option(is_call, S + dS, K, T, r, c, vol, N)
    price_down = binomial_american_option(is_call, S - dS, K, T, r, c, vol, N)
    delta = (price_up - price_down) / (2 * dS)
    
    # Gamma (Delta的偏微分)
    gamma = (price_up - 2 * base_price + price_down) / (dS ** 2)
    
    # Vega (對波動率的偏微分)
    dvol = 0.01
    price_vol_up = binomial_american_option(is_call, S, K, T, r, c, vol + dvol, N)
    vega = (price_vol_up - base_price) / dvol
    
    # Rho (對無風險利率的偏微分)
    dr = 0.0001
    price_r_up = binomial_american_option(is_call, S, K, T, r + dr, c + dr, vol, N)
    rho = (price_r_up - base_price) / dr
    
    # Theta (對時間的偏微分)
    dT = -0.01 / 365  # 減少1天
    if T + dT > 0:
        price_t_down = binomial_american_option(is_call, S, K, T + dT, r, c, vol, N)
        theta = (price_t_down - base_price) / (-dT)
    else:
        theta = 0
    
    return {
        'Value': base_price,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    }

def binomial_american_dividend(is_call, S, K, T, r, dividend_dates, dividend_amounts, vol, N):
    """
    使用二元樹方法計算含離散股利的美式選擇權價格
    
    Parameters:
    -----------
    is_call : bool
        True為買權，False為賣權
    S : float
        標的資產價格
    K : float
        履約價格
    T : float
        到期時間（年）
    r : float
        無風險利率
    dividend_dates : list
        股利發放日期（以步數表示）
    dividend_amounts : list
        股利金額
    vol : float
        波動率
    N : int
        二元樹步數
        
    Returns:
    --------
    float : 選擇權價格
    """
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    
    # 風險中性機率（無股利時）
    p = (np.exp(r * dt) - d) / (u - d)
    
    # 初始化資產價格樹
    S_tree = np.zeros((N + 1, N + 1))
    
    # 建立考慮股利的價格樹
    for i in range(N + 1):
        for j in range(i + 1):
            S_tree[j, i] = S * (u ** (i - j)) * (d ** j)
            
            # 減去已發放的股利
            for div_date, div_amount in zip(dividend_dates, dividend_amounts):
                if i >= div_date:
                    S_tree[j, i] -= div_amount
                    
            # 確保股價不為負
            S_tree[j, i] = max(0, S_tree[j, i])
    
    # 初始化選擇權價值樹
    option_tree = np.zeros((N + 1, N + 1))
    
    # 到期時的選擇權價值
    for j in range(N + 1):
        if is_call:
            option_tree[j, N] = max(0, S_tree[j, N] - K)
        else:
            option_tree[j, N] = max(0, K - S_tree[j, N])
    
    # 向後遞推
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # 歐式價值（折現期望值）
            european_value = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            
            # 提前執行價值
            if is_call:
                exercise_value = max(0, S_tree[j, i] - K)
            else:
                exercise_value = max(0, K - S_tree[j, i])
            
            # 美式選擇權價值為兩者最大值
            option_tree[j, i] = max(european_value, exercise_value)
    
    return option_tree[0, 0]
