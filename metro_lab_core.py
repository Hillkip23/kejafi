# -*- coding: utf-8 -*-
"""
metro_lab_core.py
Core Stage-1 module: data loaders, OU simulation, metro profile builder.

Fixes applied:
1. load_metro_pop pop_col detection — now checks for exact 'pop_growth' column
   first, then falls back to any column containing 'growth', then last column.
   Previously matched 'pop_1317' (raw count ~2.4M) before the decimal growth
   rate column, causing pop_growth: 2427024 in the API response.
"""

import math
from enum import Enum
from typing import Dict, Optional, TypedDict
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================
# Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "app" / "data"

ZORI_PATH      = DATA_DIR / "Zillow_Observed_Rent_Index.csv"
METRO_POP_PATH = DATA_DIR / "metro_pop_growth.csv"
METRO_ELAS_PATH = DATA_DIR / "metro_elasticity.csv"
METRO_PCI_PATH  = DATA_DIR / "metro_percapita.csv"


# ==========================
# Metro name mapping
# ==========================

ZORI_REGION_MAP: Dict[str, str] = {
    # ------------------------------------------------------------------
    # Full MSA labels (used by Stage 2 property metro field)
    # ------------------------------------------------------------------
    "Charlotte-Concord-Gastonia, NC-SC":           "Charlotte, NC",
    "New York-Newark-Jersey City, NY-NJ-PA":        "New York, NY",
    "San Francisco-Oakland-Berkeley, CA":           "San Francisco, CA",
    "Miami-Fort Lauderdale-Pompano Beach, FL":      "Miami, FL",
    "Seattle-Tacoma-Bellevue, WA":                  "Seattle, WA",
    "Atlanta-Sandy Springs-Alpharetta, GA":         "Atlanta, GA",
    "Austin-Round Rock-Georgetown, TX":             "Austin, TX",
    "Dallas-Fort Worth-Arlington, TX":              "Dallas, TX",
    "Houston-The Woodlands-Sugar Land, TX":         "Houston, TX",
    "Chicago-Naperville-Elgin, IL-IN-WI":           "Chicago, IL",
    "Denver-Aurora-Lakewood, CO":                   "Denver, CO",
    "Boston-Cambridge-Newton, MA-NH":               "Boston, MA",
    "Phoenix-Mesa-Chandler, AZ":                    "Phoenix, AZ",
    "Nashville-Davidson-Murfreesboro-Franklin, TN": "Nashville, TN",

    # ------------------------------------------------------------------
    # Short names used by 1.py selectbox and API /risk/, /stress/, etc.
    # ------------------------------------------------------------------
    "Charlotte":     "Charlotte, NC",
    "Atlanta":       "Atlanta, GA",
    "Miami":         "Miami, FL",
    "Austin":        "Austin, TX",
    "Dallas":        "Dallas, TX",
    "Houston":       "Houston, TX",
    "San Francisco": "San Francisco, CA",
    "New York":      "New York, NY",
    "Chicago":       "Chicago, IL",
    "Denver":        "Denver, CO",
    "Seattle":       "Seattle, WA",
    "Boston":        "Boston, MA",
    "Phoenix":       "Phoenix, AZ",
    "Nashville":     "Nashville, TN",
}


def normalize_metro_name(name: str) -> str:
    """
    Convert any metro label — short name, full MSA string, or ZORI
    RegionName — to the exact RegionName present in the ZORI CSV.
    Returns the input unchanged if no mapping found (passthrough for
    labels already in ZORI format like 'Charlotte, NC').
    """
    return ZORI_REGION_MAP.get(name, name)


# ==========================
# Loaders (lazy + module-level cache)
# ==========================

_zori_df   = None
_metro_pop = None
_metro_elas = None
_metro_pci  = None


def load_zori() -> pd.DataFrame:
    global _zori_df
    if _zori_df is not None:
        return _zori_df

    if not ZORI_PATH.exists():
        raise FileNotFoundError(f"ZORI data not found at {ZORI_PATH}")

    df = pd.read_csv(ZORI_PATH, header=None)

    # Promote first row to header if needed
    first_values = df.iloc[0, :5].tolist()
    if first_values[2] == "RegionName":
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

    if "RegionName" not in df.columns:
        raise ValueError("ZORI file does not contain a 'RegionName' column.")

    _zori_df = df
    return _zori_df


def load_metro_pop() -> pd.DataFrame:
    global _metro_pop
    if _metro_pop is not None:
        return _metro_pop

    if not METRO_POP_PATH.exists():
        raise FileNotFoundError(f"Metro pop data not found at {METRO_POP_PATH}")

    df = pd.read_csv(METRO_POP_PATH)

    # Auto-detect metro name column
    metro_col = None
    possible_names = ['metro', 'Metro', 'msa', 'MSA', 'name', 'Name',
                      'region', 'Region', 'msaname']
    for col in df.columns:
        if col in possible_names:
            metro_col = col
            break
    if metro_col is None:
        metro_col = df.columns[0]

    # FIX 1: detect pop_growth column — check exact name first to avoid
    # accidentally matching 'pop_1317' or 'pop_1822' (raw population counts)
    # before the decimal growth-rate column.
    pop_col = None
    if "pop_growth" in df.columns:
        # Exact match — this is the correct decimal growth rate column
        pop_col = "pop_growth"
    else:
        # Fall back: first column whose name contains 'growth'
        for col in df.columns:
            if "growth" in col.lower():
                pop_col = col
                break
    if pop_col is None:
        # Last resort: use the final column
        pop_col = df.columns[-1]

    df["metro_clean"] = df[metro_col].astype(str)
    df["pop_growth"]  = pd.to_numeric(df[pop_col], errors="coerce")
    _metro_pop = df
    return _metro_pop


def load_metro_elas() -> pd.DataFrame:
    global _metro_elas
    if _metro_elas is not None:
        return _metro_elas

    if not METRO_ELAS_PATH.exists():
        raise FileNotFoundError(f"Metro elasticity data not found at {METRO_ELAS_PATH}")

    df = pd.read_csv(METRO_ELAS_PATH)

    # Auto-detect metro name column
    metro_col = None
    for col in df.columns:
        if col.lower() in ['msaname', 'metro', 'msa', 'name', 'region']:
            metro_col = col
            break
    if metro_col is None:
        metro_col = 'msaname'

    df["metro_clean"] = (
        df[metro_col]
        .astype(str)
        .str.replace(r"\s*\(MSA\)|\s*\(PMSA\)|\s*NECMA", "", regex=True)
    )
    _metro_elas = df
    return _metro_elas


def load_metro_pci() -> pd.DataFrame:
    global _metro_pci
    if _metro_pci is not None:
        return _metro_pci

    if not METRO_PCI_PATH.exists():
        raise FileNotFoundError(f"Metro PCI data not found at {METRO_PCI_PATH}")

    df = pd.read_csv(METRO_PCI_PATH)
    cols = df.columns.tolist()
    metro_col = cols[0]

    # Find year columns by name
    year_cols = {}
    for col in cols[1:]:
        col_str = str(col).lower()
        if '2021' in col_str or '21' in col_str:
            year_cols['2021'] = col
        elif '2022' in col_str or '22' in col_str:
            year_cols['2022'] = col
        elif '2023' in col_str or '23' in col_str:
            year_cols['2023'] = col

    # Positional fallback if year detection fails
    if len(year_cols) < 3 and len(cols) >= 4:
        year_cols = {
            '2021': cols[1],
            '2022': cols[2],
            '2023': cols[3],
        }

    df = df.rename(columns={
        metro_col:                        "metro",
        year_cols.get('2021', cols[1]):   "pci_2021",
        year_cols.get('2022', cols[2]):   "pci_2022",
        year_cols.get('2023', cols[3]):   "pci_2023",
    })

    keep_cols = [c for c in ["metro", "pci_2021", "pci_2022", "pci_2023"]
                 if c in df.columns]
    df = df[keep_cols]

    df = df.dropna(subset=["metro"])
    df = df[~df["metro"].astype(str).str.contains("United States", na=False)]
    df = df[~df["metro"].astype(str).str.contains(
        "Metropolitan Statistical Areas", na=False
    )]

    for col in ["pci_2021", "pci_2022", "pci_2023"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .astype(float)
            )

    df["metro_clean"] = df["metro"].astype(str)
    _metro_pci = df
    return _metro_pci


# ==========================
# ZORI helper
# ==========================

def get_series(external_name: str) -> Optional[pd.Series]:
    """Return monthly rent index series for a given metro label."""
    region  = normalize_metro_name(external_name)
    zori_df = load_zori()
    sub = zori_df[zori_df["RegionName"] == region]

    if sub.empty:
        return None

    date_cols = [c for c in sub.columns if str(c)[:4].isdigit()]
    values = sub[date_cols].iloc[0].astype(float).dropna()
    return pd.Series(values.values, index=pd.to_datetime(values.index))


def metro_label_for_join(zori_name: str) -> str:
    return zori_name.replace("\u2013", "-")


# ==========================
# Fundamentals join
# ==========================

def get_metro_fundamentals(external_name: str) -> Dict:
    label = metro_label_for_join(external_name)
    base  = label.split(",")[0].strip()

    # PCI
    pci_2023 = None
    try:
        pci_df = load_metro_pci()
        mask   = pci_df["metro_clean"].str.contains(
            base, case=False, na=False, regex=False
        )
        pci_row = pci_df[mask]
        if not pci_row.empty:
            val = pci_row["pci_2023"].iloc[0]
            if isinstance(val, str):
                val = val.replace(",", "").replace("$", "")
            pci_2023 = float(val)
    except Exception:
        pass

    # Population growth
    pop_growth = None
    try:
        pop_df = load_metro_pop()
        mask   = pop_df["metro_clean"].str.contains(
            base, case=False, na=False, regex=False
        )
        pop_row = pop_df[mask]
        if not pop_row.empty:
            pop_growth = float(pop_row["pop_growth"].iloc[0])
    except Exception:
        pass

    # Elasticity + supply bucket
    metro_elasticity = None
    supply_bucket    = None
    try:
        elas_df = load_metro_elas()
        mask    = elas_df["metro_clean"].str.contains(
            base, case=False, na=False, regex=False
        )
        elas_row = elas_df[mask]
        if not elas_row.empty:
            metro_elasticity = float(elas_row["elasticity"].iloc[0])
            if metro_elasticity < 1.5:
                supply_bucket = "Very Inelastic"
            elif metro_elasticity < 2.5:
                supply_bucket = "Inelastic"
            elif metro_elasticity < 3.5:
                supply_bucket = "Moderate"
            else:
                supply_bucket = "Elastic"
    except Exception:
        pass

    return {
        "pci_2023":        pci_2023,
        "pop_growth":      pop_growth,
        "metro_elasticity": metro_elasticity,
        "supply_bucket":   supply_bucket,
    }


# ==========================
# OU / simulation engine
# ==========================

class EngineMode(Enum):
    STANDARD = "Standard"


class StressScenario(Enum):
    BASE_CASE = "Base Case"


def get_scenario_params(scenario: StressScenario) -> Dict:
    return {
        "sigma_mult":       1.0,
        "kappa_mult":       1.0,
        "vacancy_shock":    0.0,
        "rent_growth_shift": 0.0,
        "cap_rate_shock":   0.0,
        "jump_intensity":   0.0,
    }


def estimate_ou_from_series(logR: np.ndarray, dt_years: float) -> Dict:
    """
    Estimate Ornstein-Uhlenbeck parameters from a log rent series.

    Returns:
        Dict with kappa (mean reversion speed), theta (long-run mean),
        sigma (volatility).
    """
    x = logR[:-1]
    y = logR[1:]
    if len(x) < 5:
        return {"kappa": 0.5, "theta": float(np.mean(logR)), "sigma": 0.10}

    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - b * x.mean()
    b = float(np.clip(b, 1e-6, 0.999999))
    kappa = -math.log(b) / dt_years
    theta = a / (1.0 - b)
    resid = y - (a + b * x)
    sigma_eps = float(np.std(resid, ddof=1))
    var_factor = (
        (1.0 - math.exp(-2.0 * kappa * dt_years)) / (2.0 * kappa)
        if kappa > 1e-12
        else dt_years
    )
    sigma = sigma_eps / math.sqrt(var_factor)
    return {"kappa": float(kappa), "theta": float(theta), "sigma": float(sigma)}


def run_simulation(
    series: pd.Series,
    T_years: float,
    n_sims: int,
    units: int,
    vacancy: float,
    op_cost: float,
    discount: float,
    mode: EngineMode,
    scenario_params: Dict,
    county_adj: Dict,
    seed: int = 42,
) -> Dict:
    rent_hist = series.values
    logR      = np.log(rent_hist)
    dt_years  = 1.0 / 12.0

    ou_params = estimate_ou_from_series(logR, dt_years)
    kappa = ou_params["kappa"]
    theta = ou_params["theta"]
    sigma = ou_params["sigma"]

    sigma_adj = sigma * scenario_params.get("sigma_mult", 1.0)
    kappa_adj = kappa * scenario_params.get("kappa_mult", 1.0)
    theta_adj = theta + scenario_params.get("rent_growth_shift", 0.0)

    vacancy_adj = min(
        vacancy
        + scenario_params.get("vacancy_shock", 0.0)
        * county_adj.get("stress_factor", 1),
        0.50,
    )

    steps = int(T_years * 12)
    X0    = math.log(float(rent_hist[-1]) * 12.0)

    rng   = np.random.default_rng(seed)
    dt    = T_years / steps
    exp_k = math.exp(-kappa_adj * dt)
    ou_var = (
        (1.0 - math.exp(-2.0 * kappa_adj * dt)) / (2.0 * kappa_adj)
        if kappa_adj > 1e-12
        else dt
    )
    ou_std = sigma_adj * math.sqrt(ou_var)

    paths = np.zeros((n_sims, steps + 1))
    paths[:, 0] = X0
    for t in range(steps):
        paths[:, t + 1] = (
            theta_adj
            + (paths[:, t] - theta_adj) * exp_k
            + ou_std * rng.standard_normal(n_sims)
        )

    rent_T = np.exp(paths[:, -1])
    gross  = rent_T * units
    noi    = gross * (1 - vacancy_adj) * (1 - op_cost)
    price  = noi / ((1 + discount) ** T_years)

    var95 = float(np.quantile(price, 0.05))
    return {
        "price": price,
        "mean":  float(np.mean(price)),
        "var95": var95,
        "cvar":  float(np.mean(price[price <= var95])),
        "noi":   noi,
    }


# ==========================
# Metro profile
# ==========================

class MetroProfile(TypedDict):
    metro_name:       str
    pci_2023:         Optional[float]
    pop_growth:       Optional[float]
    metro_elasticity: Optional[float]
    supply_bucket:    Optional[str]
    yoy_rent_growth:  float
    rent_volatility:  float
    trend_label:      str
    mean_price:       float
    var95:            float
    cvar:             float
    risk_score:       float
    risk_bucket:      str


def build_metro_profile(external_name: str) -> MetroProfile:
    """
    Core Stage-1 function used by both the Metro Lab UI and Stage-2 app.
    """
    series = get_series(external_name)
    if series is None:
        raise ValueError(
            f"No ZORI series for '{external_name}'. "
            "Check ZORI_REGION_MAP for a mapping or verify the RegionName."
        )

    fundamentals = get_metro_fundamentals(external_name)

    # Year-on-year rent growth
    if len(series) > 12:
        yoy = series.iloc[-1] / series.iloc[-13] - 1
    else:
        yoy = 0.05

    # Annualised rent volatility
    rets = series.pct_change().dropna()
    vol  = float(rets.std() * np.sqrt(12)) if len(rets) > 0 else 0.08

    if yoy > 0.08:
        trend = "accelerating"
    elif yoy < 0.02:
        trend = "decelerating"
    else:
        trend = "stable"

    base_params = get_scenario_params(StressScenario.BASE_CASE)
    county_adj  = {"elasticity": 0.3, "stress_factor": 1.0}

    res = run_simulation(
        series,
        T_years=2.0,
        n_sims=5000,
        units=100,
        vacancy=0.05,
        op_cost=0.35,
        discount=0.055,
        mode=EngineMode.STANDARD,
        scenario_params=base_params,
        county_adj=county_adj,
        seed=123,
    )

    tail_ratio = (res["mean"] - res["cvar"]) / max(res["mean"], 1e-6)
    risk_score = max(0.0, 1.0 - tail_ratio) * 100.0

    if risk_score >= 80:
        bucket = "Low"
    elif risk_score >= 60:
        bucket = "Moderate"
    elif risk_score >= 40:
        bucket = "Elevated"
    else:
        bucket = "High"

    return {
        "metro_name":       external_name,
        "pci_2023":         fundamentals["pci_2023"],
        "pop_growth":       fundamentals["pop_growth"],
        "metro_elasticity": fundamentals["metro_elasticity"],
        "supply_bucket":    fundamentals["supply_bucket"],
        "yoy_rent_growth":  float(yoy),
        "rent_volatility":  float(vol),
        "trend_label":      trend,
        "mean_price":       res["mean"],
        "var95":            res["var95"],
        "cvar":             res["cvar"],
        "risk_score":       risk_score,
        "risk_bucket":      bucket,
    }


# ==========================
# Explicit exports
# ==========================

__all__ = [
    "load_zori",
    "load_metro_pci",
    "load_metro_elas",
    "load_metro_pop",
    "normalize_metro_name",
    "metro_label_for_join",
    "estimate_ou_from_series",
    "run_simulation",
    "build_metro_profile",
    "MetroProfile",
    "EngineMode",
    "StressScenario",
    "get_scenario_params",
    "DATA_DIR",
]