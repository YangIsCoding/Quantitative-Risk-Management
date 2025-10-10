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

def super_efficient_portfolio(expected_rts,cov,rf=0.0425):
    """Given a target return, use assets to find the optimal portfolio with lowest risk"""
    fun=lambda wts: -(wts@expected_rts-rf)/np.sqrt(wts@cov@wts)
    x0 = np.full(expected_rts.shape[0],1/expected_rts.shape[0])
    cons = [{'type':'ineq', 'fun':lambda x:x},
        {'type':'eq', 'fun':lambda x:sum(x)-1}]
    bounds = [(0, 1) for _ in range(expected_rts.shape[0])]
    res = optimize.minimize(fun, x0, method='SLSQP',bounds=bounds,constraints=cons)
    return res

def RiskParity(cov):
    """Given a target return, use assets to find the optimal portfolio with lowest risk"""
    fun=lambda w: (w*(cov@w)/np.sqrt(w@cov@w)).std()
    x0 = np.full(cov.shape[0],1/cov.shape[0])
    cons = [{'type':'ineq', 'fun':lambda x:x},
        {'type':'eq', 'fun':lambda x:sum(x)-1}]
    bounds = [(0, 1) for _ in range(cov.shape[0])]
    res = optimize.minimize(fun, x0, method='SLSQP',bounds=bounds,constraints=cons)
    return res
    
def riskBudget(w,cov):
    """Calculate the portion of risk each stock of portfolio has. The sum of result is 1"""
    portfolioStd=np.sqrt(w@cov@w)
    Csd=w*(cov@w)/portfolioStd
    return Csd/portfolioStd

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

class NotPsdError(Exception):
    """ 
    Used for expection raise if the input matrix is not sysmetric positive definite 
    """
    pass

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

class NegativeEigError(Exception):
    """ 
    Used for expection raise if matrix has the negative eigvalue
    """
    pass

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