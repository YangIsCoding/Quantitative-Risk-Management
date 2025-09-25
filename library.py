import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings

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
    Find the nearest positive semi-definite matrix using eigenvalue decomposition
    """
    # Ensure the matrix is symmetric
    A = (matrix + matrix.T) / 2
    
    # Eigenvalue decomposition
    eigenvals, eigenvecs = np.linalg.eigh(A)
    
    # Set negative eigenvalues to epsilon (typically 0)
    eigenvals = np.maximum(eigenvals, epsilon)
    
    # Reconstruct the matrix
    result = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    return result

def higham_nearestPSD(A, maxIts=100, tol=1e-8):
    """
    Higham's 2002 algorithm for finding the nearest PSD matrix
    """
    n = A.shape[0]
    
    # Ensure matrix is symmetric
    A = (A + A.T) / 2
    
    # Initialize
    Y = A.copy()
    Delta_S = np.zeros_like(A)
    
    for k in range(maxIts):
        # Step 1: Project onto S (symmetric matrices with 0 diagonal)
        # For general matrices, this step is skipped
        
        # Step 2: Project onto positive semidefinite cone
        eigenvals, eigenvecs = np.linalg.eigh(Y - Delta_S)
        eigenvals_pos = np.maximum(eigenvals, 0)
        X = eigenvecs @ np.diag(eigenvals_pos) @ eigenvecs.T
        
        # Step 3: Update Delta_S
        Delta_S = X - (Y - Delta_S)
        
        # Step 4: Project back to original constraint set
        Y_new = X
        
        # Check convergence
        if np.linalg.norm(Y - Y_new, 'fro') <= tol:
            break
        
        Y = Y_new
    
    return Y

def chol_psd(matrix):
    """
    Cholesky decomposition that handles PSD matrices
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        # If not PD, find nearest PSD and decompose
        psd_matrix = near_psd(matrix)
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