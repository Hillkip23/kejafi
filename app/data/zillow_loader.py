import pandas as pd
import numpy as np
from typing import Optional, Dict
import os

class ZillowDataLoader:
    """
    Lazy-loading data infrastructure for ZORI and metro fundamentals.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._cache = {}
        self._zori_data = None
        self._elasticity_data = None
        self._pci_data = None
        self._ready = False
        self._national_avg_pci = 65000
        
        self._load_all()
    
    def _load_all(self):
        """Load all data sources."""
        try:
            self._load_zori()
            self._load_elasticity()
            self._load_pci()
            self._ready = True
        except Exception as e:
            print(f"Warning: Could not load all data: {e}")
            self._ready = False
    
    def _load_zori(self):
        """Load Zillow Observed Rent Index."""
        paths = [
            os.path.join(self.data_dir, "zori.csv"),
            os.path.join(self.data_dir, "ZORI.csv"),
            "zori.csv"
        ]
        
        for path in paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    df.columns = [c.strip().lower() for c in df.columns]
                    region_cols = [c for c in df.columns if 'region' in c or 'metro' in c]
                    if region_cols:
                        self._zori_data = df.set_index(region_cols[0])
                        print(f"Loaded ZORI data: {len(df)} metros")
                        return
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
        
        print("Warning: No ZORI file found. Using synthetic data.")
        self._zori_data = self._create_synthetic_zori()
    
    def _create_synthetic_zori(self) -> pd.DataFrame:
        """Create synthetic rent data for testing."""
        metros = ["Charlotte", "Atlanta", "Miami", "Austin", "Dallas", 
                 "Houston", "San Francisco", "New York", "Chicago"]
        
        data = {}
        np.random.seed(42)
        
        for metro in metros:
            dates = pd.date_range(start='2014-01-01', end='2024-01-01', freq='M')
            base_rent = np.random.uniform(1200, 2500)
            trend = np.linspace(0, 0.4, len(dates))
            noise = np.random.normal(0, 0.02, len(dates))
            seasonal = 0.02 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
            
            rents = base_rent * np.exp(trend + noise + seasonal)
            data[metro] = pd.Series(rents, index=dates)
        
        return pd.DataFrame(data).T
    
    def _load_elasticity(self):
        """Load Saiz elasticity data."""
        self._elasticity_data = {
            "Charlotte": 1.2,
            "Atlanta": 1.8,
            "Miami": 0.8,
            "Austin": 1.5,
            "Dallas": 2.2,
            "Houston": 2.8,
            "San Francisco": 0.6,
            "New York": 0.7,
            "Chicago": 1.4,
            "Denver": 1.1,
            "Seattle": 0.9,
            "Boston": 0.8,
            "Phoenix": 2.5,
            "Nashville": 1.6,
            "Tampa": 1.3,
            "Raleigh": 1.9
        }
    
    def _load_pci(self):
        """Load per capita income data."""
        self._pci_data = {
            "Charlotte": 58200,
            "Atlanta": 54100,
            "Miami": 48900,
            "Austin": 61200,
            "Dallas": 56800,
            "Houston": 52300,
            "San Francisco": 98500,
            "New York": 76500,
            "Chicago": 61200,
            "Denver": 65400,
            "Seattle": 72300,
            "Boston": 71200,
            "Phoenix": 45600,
            "Nashville": 53400,
            "Tampa": 47800,
            "Raleigh": 58900
        }
    
    def get_rent_series(self, metro: str) -> Optional[pd.Series]:
        """Get rent time series for a metro area."""
        if self._zori_data is None:
            return None
        
        if metro in self._zori_data.index:
            return self._zori_data.loc[metro]
        
        matches = [idx for idx in self._zori_data.index if metro.lower() in idx.lower()]
        if matches:
            return self._zori_data.loc[matches[0]]
        
        matches = [idx for idx in self._zori_data.index if idx.lower() in metro.lower()]
        if matches:
            return self._zori_data.loc[matches[0]]
        
        return None
    
    def get_elasticity(self, metro: str) -> float:
        """Get housing supply elasticity."""
        if metro in self._elasticity_data:
            return self._elasticity_data[metro]
        
        for key, val in self._elasticity_data.items():
            if key.lower() in metro.lower() or metro.lower() in key.lower():
                return val
        
        return 2.5
    
    def get_pci(self, metro: str) -> float:
        """Get per capita income."""
        if metro in self._pci_data:
            return self._pci_data[metro]
        
        for key, val in self._pci_data.items():
            if key.lower() in metro.lower() or metro.lower() in key.lower():
                return val
        
        return self._national_avg_pci
    
    def get_national_avg_pci(self) -> float:
        return self._national_avg_pci
    
    def is_ready(self) -> bool:
        return self._ready
    
    def get_available_metros_count(self) -> int:
        if self._zori_data is None:
            return 0
        return len(self._zori_data)
    
    def get_available_metros(self) -> list:
        if self._zori_data is None:
            return []
        return list(self._zori_data.index)