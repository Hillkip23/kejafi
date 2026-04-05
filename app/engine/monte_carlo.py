import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats
from app.engine.ou_estimator import OrnsteinUhlenbeckEstimator

class MonteCarloRiskEngine:
    """
    Monte Carlo simulation for real estate risk metrics.
    
    5,000 paths, VaR/CVaR calculation, terminal value distributions.
    """
    
    def __init__(self, n_simulations: int = 5000):
        self.n_sims = n_simulations
        self.ou = OrnsteinUhlenbeckEstimator()
        
    def calculate_var(self,
                      rent_series: pd.Series,
                      horizon_years: float = 1.0,
                      confidence: float = 0.95,
                      annual_rent_to_value: float = 0.055,
                      ou_params: Optional[Dict] = None) -> Dict:
        """
        Calculate VaR, CVaR, and expected returns via Monte Carlo.
        
        Args:
            rent_series: Historical rent data
            horizon_years: Investment horizon
            confidence: VaR confidence level (e.g., 0.95)
            annual_rent_to_value: Cap rate for valuation
            ou_params: Pre-computed OU parameters (optional)
            
        Returns:
            Risk metrics dictionary
        """
        # Get OU parameters if not provided
        if ou_params is None:
            ou_params = self.ou.estimate(rent_series)
            
        X0 = ou_params["current_log_rent"]
        current_rent = ou_params["current_rent"]
        
        # Current property value (simple income approach)
        annual_rent = current_rent * 12
        current_value = annual_rent / annual_rent_to_value
        
        # Simulate 5,000 paths
        n_steps = int(horizon_years * 12)
        terminal_values = []
        
        for i in range(self.n_sims):
            path = self.ou.simulate_path(ou_params, X0, horizon_years, n_steps)
            terminal_rent = np.exp(path[-1])
            terminal_annual_rent = terminal_rent * 12
            terminal_value = terminal_annual_rent / annual_rent_to_value
            terminal_values.append(terminal_value)
            
        terminal_values = np.array(terminal_values)
        
        # Calculate returns
        returns = (terminal_values - current_value) / current_value
        
        # Sort for quantile calculations
        sorted_returns = np.sort(returns)
        
        # VaR and CVaR
        var_idx = int((1 - confidence) * len(sorted_returns))
        var_threshold = sorted_returns[var_idx]
        cvar = np.mean(sorted_returns[:var_idx+1])  # Expected shortfall
        
        return {
            "current_value": float(current_value),
            "expected_terminal_value": float(np.mean(terminal_values)),
            "expected_return": float(np.mean(returns)),
            "volatility": float(np.std(returns)),
            f"var_{int(confidence*100)}": float(var_threshold),
            f"cvar_{int(confidence*100)}": float(cvar),
            "percentile_5": float(np.percentile(returns, 5)),
            "percentile_95": float(np.percentile(returns, 95)),
            "probability_positive": float(np.mean(returns > 0)),
            "median_return": float(np.median(returns)),
            "skewness": float(stats.skew(returns)) if len(returns) > 8 else 0.0,
            "kurtosis": float(stats.kurtosis(returns)) if len(returns) > 8 else 0.0,
            "current_rent": float(current_rent)
        }