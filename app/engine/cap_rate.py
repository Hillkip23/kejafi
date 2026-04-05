class DynamicCapRateModel:
    """
    Three-factor cap rate model with elasticity and PCI adjustments.
    
    c(m) = c_base(m) + delta_elasticity + delta_PCI
    """
    
    # Elasticity buckets from Saiz (2010)
    ELASTICITY_BUCKETS = {
        "VERY_INELASTIC": (0.0, 1.5, -0.0035),      # -35 bps
        "INELASTIC": (1.5, 2.5, -0.0025),           # -25 bps
        "MODERATE": (2.5, 3.5, 0.0),                # 0 bps
        "ELASTIC": (3.5, float('inf'), 0.0015)      # +15 bps
    }
    
    # Base cap rates by major metro
    BASE_CAP_RATES = {
        "Charlotte": 0.055,
        "Atlanta": 0.052,
        "Miami": 0.050,
        "Austin": 0.045,
        "Dallas": 0.048,
        "Houston": 0.050,
        "San Francisco": 0.040,
        "New York": 0.045,
        "Chicago": 0.055,
        "Denver": 0.046,
        "Seattle": 0.042,
        "Boston": 0.043,
        "Washington": 0.044,
        "Phoenix": 0.048,
        "Nashville": 0.050,
        "Tampa": 0.052,
        "Orlando": 0.051,
        "Raleigh": 0.048,
        "Charleston": 0.053
    }
    
    DEFAULT_CAP_RATE = 0.055  # 5.5% national average
    
    def calculate(self,
                  metro: str,
                  elasticity: float,
                  pci: float,
                  national_avg_pci: float = 65000) -> dict:
        """
        Calculate dynamic cap rate.
        
        Args:
            metro: Metropolitan area name
            elasticity: Housing supply elasticity (Saiz index)
            pci: Per capita income
            national_avg_pci: National average PCI for comparison
            
        Returns:
            Cap rate components and final rate
        """
        # Base rate lookup
        c_base = self.BASE_CAP_RATES.get(metro, self.DEFAULT_CAP_RATE)
        
        # Elasticity adjustment
        delta_elasticity = self._get_elasticity_adjustment(elasticity)
        bucket = self._get_bucket_name(elasticity)
        
        # PCI adjustment: -20 bps per 30% deviation from national avg
        if national_avg_pci > 0:
            pci_ratio = (pci - national_avg_pci) / national_avg_pci
            pci_ratio_clipped = max(-0.3, min(0.3, pci_ratio))  # Clip to ±30%
            delta_pci = -0.0020 * pci_ratio_clipped
        else:
            delta_pci = 0.0
        
        # Final cap rate with bounds
        cap_rate = c_base + delta_elasticity + delta_pci
        cap_rate = max(0.03, min(0.09, cap_rate))  # 3%-9% bounds
        
        return {
            "metro": metro,
            "base_rate": round(c_base, 4),
            "elasticity": round(elasticity, 2) if elasticity else None,
            "elasticity_bucket": bucket,
            "elasticity_adjustment": round(delta_elasticity, 4),
            "pci": round(pci, 2) if pci else None,
            "pci_adjustment": round(delta_pci, 4),
            "final_cap_rate": round(cap_rate, 4),
            "implied_multiplier": round(1 / cap_rate, 2),
            "cap_rate_bps": int(cap_rate * 10000)
        }
    
    def _get_elasticity_adjustment(self, elasticity: float) -> float:
        """Get basis point adjustment based on elasticity bucket."""
        if elasticity is None:
            return 0.0
        for bucket, (low, high, adj) in self.ELASTICITY_BUCKETS.items():
            if low <= elasticity < high:
                return adj
        return 0.0
    
    def _get_bucket_name(self, elasticity: float) -> str:
        """Classify elasticity into bucket."""
        if elasticity is None:
            return "UNKNOWN"
        for bucket, (low, high, _) in self.ELASTICITY_BUCKETS.items():
            if low <= elasticity < high:
                return bucket
        return "UNKNOWN"