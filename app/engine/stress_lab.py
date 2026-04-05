import numpy as np
from typing import Dict
from app.engine.ou_estimator import OrnsteinUhlenbeckEstimator

class StressLab:
    """
    Stress testing with jump-diffusion and scenario analysis.
    """
    
    # Scenario parameters from paper Table 3
    SCENARIOS = {
        "BASE_CASE": {
            "lambda": 0.0,
            "mu_j": 0.0,
            "sigma_j": 0.0,
            "kappa_mult": 1.0,
            "theta_shift": 0.0,
            "description": "Normal market conditions"
        },
        "COVID_SHOCK": {
            "lambda": 0.3,
            "mu_j": -0.30,
            "sigma_j": 0.20,
            "kappa_mult": 1.2,
            "theta_shift": -0.05,
            "description": "Pandemic-style demand shock with rent declines"
        },
        "GFC_2008": {
            "lambda": 0.5,
            "mu_j": -0.20,
            "sigma_j": 0.25,
            "kappa_mult": 0.8,
            "theta_shift": -0.10,
            "description": "Credit crunch with prolonged recovery"
        },
        "STAGFLATION": {
            "lambda": 0.2,
            "mu_j": 0.10,
            "sigma_j": 0.15,
            "kappa_mult": 1.5,
            "theta_shift": 0.15,
            "description": "Supply shocks with inflationary pressure"
        }
    }
    
    def __init__(self, n_simulations: int = 5000):
        self.n_sims = n_simulations
        self.ou = OrnsteinUhlenbeckEstimator()
        
    def run_scenario(self,
                     rent_series,
                     ou_params: Dict,
                     scenario: str,
                     n_sims: int = 5000) -> Dict:
        """
        Run stress scenario with jump-diffusion.
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
            
        params = self.SCENARIOS[scenario]
        
        # Adjust OU parameters for scenario
        stressed_params = ou_params.copy()
        stressed_params["kappa"] *= params["kappa_mult"]
        stressed_params["theta"] += params["theta_shift"]
        
        # Simulate with jumps
        X0 = ou_params["current_log_rent"]
        T = 1.0
        n_steps = 12
        
        terminal_values = []
        
        for _ in range(n_sims):
            path = self._simulate_with_jumps(
                stressed_params, 
                X0, 
                T, 
                n_steps,
                jump_intensity=params["lambda"],
                jump_mean=params["mu_j"],
                jump_std=params["sigma_j"]
            )
            
            terminal_rent = np.exp(path[-1])
            terminal_value = terminal_rent * 12 / 0.055
            terminal_values.append(terminal_value)
            
        terminal_values = np.array(terminal_values)
        current_value = np.exp(X0) * 12 / 0.055
        returns = (terminal_values - current_value) / current_value
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        return {
            "scenario": scenario,
            "expected_return": float(np.mean(returns)),
            "volatility": float(np.std(returns)),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "probability_loss": float(np.mean(returns < 0)),
            "probability_severe_loss": float(np.mean(returns < -0.20)),
            "severity_ratio": float(abs(var_95) / np.std(returns)) if np.std(returns) > 0 else 0,
            "median_return": float(np.median(returns)),
            "jump_adjustment": params["mu_j"] if params["lambda"] > 0 else None
        }
    
    def _simulate_with_jumps(self,
                            params: Dict,
                            X0: float,
                            T: float,
                            n_steps: int,
                            jump_intensity: float,
                            jump_mean: float,
                            jump_std: float) -> np.ndarray:
        """
        Simulate OU with Merton jump-diffusion.
        """
        kappa = params["kappa"]
        theta = params["theta"]
        sigma = params["sigma"]
        
        delta = T / n_steps
        path = np.zeros(n_steps + 1)
        path[0] = X0
        
        for t in range(n_steps):
            if kappa > 0:
                exp_term = np.exp(-kappa * delta)
                mean_term = theta * (1 - exp_term)
                var_term = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * delta))
                std_term = np.sqrt(max(var_term, 0))
                
                X_next = path[t] * exp_term + mean_term + std_term * np.random.normal()
            else:
                X_next = path[t] + sigma * np.sqrt(delta) * np.random.normal()
            
            if jump_intensity > 0:
                n_jumps = np.random.poisson(jump_intensity * delta)
                if n_jumps > 0:
                    jump_size = np.random.normal(jump_mean, max(jump_std, 0.01)) * n_jumps
                    X_next += jump_size
            
            path[t + 1] = X_next
            
        return path
    
    def interpret_results(self, results: Dict, scenario: str) -> Dict:
        """Generate human-readable interpretation."""
        scenario_info = self.SCENARIOS[scenario]
        
        return {
            "scenario_description": scenario_info["description"],
            "risk_level": self._classify_risk(results),
            "key_takeaway": self._generate_takeaway(results),
            "investor_guidance": self._generate_guidance(results)
        }
    
    def _classify_risk(self, results: Dict) -> str:
        var = results["var_95"]
        if var < -0.30:
            return "SEVERE"
        elif var < -0.15:
            return "HIGH"
        elif var < -0.05:
            return "MODERATE"
        else:
            return "LOW"
    
    def _generate_takeaway(self, results: Dict) -> str:
        prob_loss = results["probability_loss"]
        var = results["var_95"]
        
        if prob_loss > 0.4:
            return f"High probability of loss ({prob_loss:.0%}), with 5% chance of losing {abs(var):.0%} or more."
        elif var < -0.10:
            return f"Moderate downside risk: 5% chance of {abs(var):.0%} loss, but positive expected return."
        else:
            return "Limited downside risk with attractive expected returns."
    
    def _generate_guidance(self, results: Dict) -> str:
        risk = self._classify_risk(results)
        if risk == "SEVERE":
            return "Consider defensive positioning or liquidity buffers. Avoid leverage."
        elif risk == "HIGH":
            return "Appropriate for risk-tolerant investors. Monitor quarterly."
        elif risk == "MODERATE":
            return "Suitable for diversified portfolios. Standard due diligence."
        else:
            return "Favorable risk profile. Can support higher leverage if desired."