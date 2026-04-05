import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class OrnsteinUhlenbeckEstimator:
    """
    Exact MLE estimation of OU parameters from rent time series.
    
    dX_t = kappa(theta - X_t)dt + sigma*dW_t
    
    where X_t = log(rent_t)
    """
    
    def __init__(self, min_observations: int = 12):
        self.min_obs = min_observations
        
    def estimate(self, rent_series: pd.Series) -> Dict:
        """
        Estimate OU parameters via AR(1) representation.
        
        Args:
            rent_series: Monthly rent observations (Level, not log)
            
        Returns:
            Dictionary with kappa, theta, sigma, half_life, etc.
        """
        if len(rent_series) < self.min_obs:
            raise ValueError(f"Need {self.min_obs}+ observations, got {len(rent_series)}")
            
        # Log transform: X_t = log(R_t)
        X = np.log(rent_series.dropna())
        n = len(X)
        
        if n < self.min_obs:
            raise ValueError(f"After dropping NA, need {self.min_obs}+ observations")
        
        # AR(1) regression: X_{t+1} = a + b*X_t + epsilon
        X_lag = X[:-1].values
        X_curr = X[1:].values
        
        # OLS estimation
        slope, intercept, r_value, p_value, std_err = stats.linregress(X_lag, X_curr)
        
        # Numerical safeguards (exact bounds from paper)
        b_hat = np.clip(slope, 1e-6, 0.999999)
        a_hat = intercept
        
        # Convert to OU parameters (Equations 2-4 from paper)
        delta = 1/12  # Monthly data in years
        kappa = -np.log(b_hat) / delta
        theta = a_hat / (1 - b_hat)
        
        # Residual standard deviation
        residuals = X_curr - (a_hat + b_hat * X_lag)
        sigma_eps = np.std(residuals, ddof=1)
        
        # OU sigma (Equation 4)
        if kappa > 0:
            sigma = sigma_eps / np.sqrt((1 - b_hat**2) / (2 * kappa))
        else:
            sigma = sigma_eps / np.sqrt(delta)  # Fallback
        
        # Feller condition: 2*kappa*theta > sigma^2 (for positivity)
        feller_satisfied = (2 * kappa * theta > sigma**2) if (kappa > 0 and theta > 0) else True
        
        return {
            "kappa": float(kappa),
            "theta": float(theta),
            "sigma": float(sigma),
            "half_life": float(np.log(2) / kappa) if kappa > 0 else float('inf'),
            "r_squared": float(r_value**2),
            "feller_satisfied": bool(feller_satisfied),
            "n_observations": int(n),
            "current_log_rent": float(X.iloc[-1]),
            "current_rent": float(np.exp(X.iloc[-1])),
            "p_value": float(p_value)
        }
    
    def simulate_path(self, 
                     params: Dict, 
                     X0: float, 
                     T: float = 1.0, 
                     n_steps: int = 12,
                     random_seed: Optional[int] = None) -> np.ndarray:
        """
        Exact discrete-time simulation of OU process.
        
        Uses analytic solution for conditional distribution.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        kappa = params["kappa"]
        theta = params["theta"]
        sigma = params["sigma"]
        
        delta = T / n_steps
        paths = np.zeros(n_steps + 1)
        paths[0] = X0
        
        if kappa <= 0:
            # Random walk approximation if mean reversion fails
            for t in range(n_steps):
                paths[t + 1] = paths[t] + sigma * np.sqrt(delta) * np.random.normal()
            return paths
        
        # Pre-compute constants for exact simulation
        exp_term = np.exp(-kappa * delta)
        mean_term = theta * (1 - exp_term)
        var_term = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * delta))
        std_term = np.sqrt(max(var_term, 0))
        
        for t in range(n_steps):
            paths[t + 1] = paths[t] * exp_term + mean_term + std_term * np.random.normal()
            
        return paths