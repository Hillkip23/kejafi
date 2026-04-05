# -*- coding: utf-8 -*-
"""
Kejafi Single-Asset Tokenization Platform
Stage 2: Select a property, run valuation with metro fundamentals, publish to API.

Fixes applied:
1. pop_growth None guard  — full_valuation no longer throws TypeError when
   metro profile is unavailable and pop_growth is None.
2. IRR calculation        — replaced CAGR proxy with a proper Newton-Raphson
   IRR solve on the full cash-flow series (initial equity outflow + annual
   levered cash flows + exit equity proceeds).
3. API None fields        — push_property_to_api coerces None numeric fields
   to safe defaults before POST so FastAPI never receives null numerics.
4. resolve_metro_label_for_property caching — wrapped in @st.cache_data so
   load_zori() is only called once per session rather than on every re-render.
5. session_state key namespacing — keys renamed to "stage2_valuation",
   "stage2_property", "stage2_market" to avoid cross-page collisions.
6. rent_volatility noted  — added comment documenting that it is available in
   the market dict but not consumed by Stage 2 valuation (reserved for Stage 3
   Monte Carlo extension).
7. Stage 1 integration    — imports export_metro_for_stage2 from research_engine
   and adds UI to pull metro risk data from Stage 1 session state.
"""

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
import os

# Stage 1 integration: Import the export pipeline
try:
    from research_engine import export_metro_for_stage2
    HAS_STAGE1 = True
except ImportError:
    HAS_STAGE1 = False

from metro_lab_core import build_metro_profile, load_zori  # Stage 1 helpers

# =========================================================
# Setup
# =========================================================

st.set_page_config(page_title="Kejafi Single-Asset Tokenization", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"

FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# =========================================================
# Dynamic metro mapping helper
# =========================================================

@st.cache_data
def resolve_metro_label_for_property(raw_metro_label: str) -> str:
    """
    Map a human/MSA-style label like 'Charlotte-Concord-Gastonia, NC-SC'
    to the actual RegionName present in the ZORI CSV.
    Lookup order: exact match → fuzzy city-name prefix → original label.
    """
    zori_df = load_zori()

    # 1) Exact match on RegionName
    if (zori_df["RegionName"] == raw_metro_label).any():
        return raw_metro_label

    # 2) Fuzzy match: RegionName contains the first city token
    base = raw_metro_label.split(",")[0].split("\u2013")[0].split("-")[0].strip()
    candidates = zori_df[zori_df["RegionName"].str.contains(base, na=False)]

    if not candidates.empty:
        if "RegionType" in candidates.columns:
            msa_rows = candidates[
                candidates["RegionType"].astype(str).str.lower().str.contains("msa")
            ]
            if not msa_rows.empty:
                return msa_rows["RegionName"].iloc[0]
        return candidates["RegionName"].iloc[0]

    # 3) Fallback: return original (build_metro_profile will raise descriptively)
    return raw_metro_label


# =========================================================
# IRR solver (FIX 2)
# =========================================================

def compute_irr(cash_flows: List[float], guess: float = 0.10) -> float:
    """
    Newton-Raphson IRR solve on an arbitrary cash-flow series.
    cash_flows[0] must be the initial outflow (negative).
    Returns IRR as a decimal (e.g. 0.12 for 12%).
    Falls back to CAGR approximation if Newton-Raphson does not converge.
    """
    if len(cash_flows) < 2:
        return 0.0

    rate = guess
    for _ in range(1000):
        npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
        dnpv = sum(
            -t * cf / (1 + rate) ** (t + 1)
            for t, cf in enumerate(cash_flows)
            if t > 0
        )
        if abs(dnpv) < 1e-12:
            break
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < 1e-8:
            rate = new_rate
            break
        rate = new_rate

    # Sanity check: reject nonsensical results and fall back to CAGR
    if not (-0.5 < rate < 5.0):
        # CAGR fallback: total return ^ (1/n) - 1
        total_out = abs(cash_flows[0])
        total_in = sum(cf for cf in cash_flows[1:] if cf > 0)
        n = len(cash_flows) - 1
        rate = (total_in / max(total_out, 1e-6)) ** (1 / max(n, 1)) - 1

    return float(rate)


# =========================================================
# Example Property Database
# =========================================================

EXAMPLE_PROPERTIES = {
    "charlotte_mfk_001": {
        "address": "1421 N Caldwell St, Charlotte, NC 28206",
        "county": "Mecklenburg County, NC",
        "metro": "Charlotte-Concord-Gastonia, NC-SC",
        "property_type": "Multifamily",
        "units": 4,
        "beds": 2,
        "baths": 2,
        "sqft": 3200,
        "year_built": 2018,
        "lot_size": 0.25,
        "list_price": 685000,
        "current_rent": 4800,
        "market_rent": 5200,
        "occupancy": 1.0,
        "property_taxes": 4200,
        "insurance": 2400,
        "maintenance_reserve": 3000,
        "management_fee": 0.08,
        "photos": [],
        "walk_score": 78,
        "transit_score": 65,
        "school_rating": 7,
        "crime_index": "B-",
        "description": (
            "Modern 4-unit multifamily in NoDa arts district. Fully occupied, "
            "recent renovation, strong rental demand."
        ),
    },
    # Add more properties here as needed.
}


# =========================================================
# Valuation Engine
# =========================================================

@dataclass
class ValuationResult:
    list_price: float
    gross_rent: float
    noi: float
    cap_rate: float
    value_income: float
    value_per_door: float
    cash_on_cash: float
    irr_projected: float
    equity_multiple: float


def cap_rate_from_metro(prop: Dict, market: Dict) -> float:
    """
    Cap rate model:
    - Base by metro name.
    - Adjust by supply elasticity bucket.
    - Adjust by PCI vs US metro median PCI.
    """
    base = 0.055
    metro = prop["metro"]

    if "Miami" in metro:
        base = 0.050
    elif "Austin" in metro:
        base = 0.045
    elif "Charlotte" in metro:
        base = 0.055

    bucket = (market.get("supply_bucket") or "").lower()
    pci = market.get("pci_2023")

    if bucket == "very inelastic":
        base -= 0.0035
    elif bucket == "inelastic":
        base -= 0.0025
    elif bucket == "elastic":
        base += 0.0015

    us_metro_pci = 72275.0
    if pci is not None:
        rel = (pci - us_metro_pci) / us_metro_pci
        base -= 0.002 * float(np.clip(rel, -0.3, 0.3))

    return max(0.03, min(base, 0.09))


class PropertyValuation:
    def __init__(self, property_data: Dict, market_data: Dict):
        self.prop = property_data
        self.market = market_data

    def calculate_noi(self, use_market_rent: bool = False) -> float:
        monthly_rent = (
            self.prop["market_rent"] if use_market_rent else self.prop["current_rent"]
        )
        annual_rent = monthly_rent * 12 * self.prop["occupancy"]
        management = annual_rent * self.prop["management_fee"]
        opex = (
            self.prop["property_taxes"]
            + self.prop["insurance"]
            + self.prop["maintenance_reserve"]
            + management
        )
        return annual_rent - opex

    def full_valuation(
        self,
        leverage: float = 0.65,
        interest_rate: float = 0.075,
        hold_period: int = 5,
    ) -> ValuationResult:
        noi_stabilized = self.calculate_noi(use_market_rent=True)
        cap_rate = cap_rate_from_metro(self.prop, self.market)
        value_income = noi_stabilized / cap_rate
        value_per_door = value_income / self.prop["units"]

        loan_amount = self.prop["list_price"] * leverage
        equity = self.prop["list_price"] - loan_amount
        annual_debt_service = loan_amount * interest_rate  # interest-only

        zori_yoy = self.market["yoy_rent_growth"]

        
        pop_g: Optional[float] = self.market.get("pop_growth")
        growth_adj = 0.0
        if pop_g is not None:
            if pop_g > 0.08:
                growth_adj = 0.01
            elif pop_g < 0:
                growth_adj = -0.01

        growth = max(-0.02, zori_yoy + growth_adj)

        # Build levered annual cash flows
        annual_cash_flows: List[float] = []
        for year in range(1, hold_period + 1):
            noi_growth = noi_stabilized * ((1 + growth) ** year)
            cf = noi_growth - annual_debt_service
            annual_cash_flows.append(cf)

        exit_noi = annual_cash_flows[-1] + annual_debt_service
        exit_cap = cap_rate + 0.005
        exit_value = exit_noi / exit_cap
        loan_balance = loan_amount
        exit_equity = exit_value - loan_balance

        # Add exit proceeds to the final year cash flow
        terminal_cash_flows = annual_cash_flows[:-1] + [
            annual_cash_flows[-1] + exit_equity
        ]

        total_cash = sum(annual_cash_flows) + exit_equity
        equity_multiple = total_cash / max(equity, 1e-6)

        irr_cash_flows = [-equity] + terminal_cash_flows
        irr = compute_irr(irr_cash_flows)

        cash_on_cash = annual_cash_flows[0] / max(equity, 1e-6)

        return ValuationResult(
            list_price=self.prop["list_price"],
            gross_rent=self.prop["current_rent"] * 12,
            noi=noi_stabilized,
            cap_rate=cap_rate,
            value_income=value_income,
            value_per_door=value_per_door,
            cash_on_cash=cash_on_cash,
            irr_projected=irr,
            equity_multiple=equity_multiple,
        )


# =========================================================
# FastAPI publish helper
# =========================================================

def _coerce(value, default=0.0):
    """Return value if not None, else default. Prevents null numerics in POST."""
    return value if value is not None else default


def push_property_to_api(
    prop_id: str,
    valuation: ValuationResult,
    market: Dict,
    prop: Dict,
):
    tokenized_equity_fraction = 0.8
    total_supply = 100_000

    nav_total = valuation.value_income * tokenized_equity_fraction
    nav_per_token = nav_total / total_supply if total_supply > 0 else 0.0
    initial_price = nav_per_token
    seed_tokens = int(total_supply * 0.2)
    seed_quote = seed_tokens * initial_price

    payload = {
        "id": prop_id,

        # Core property info
        "address": prop["address"],
        "county": prop["county"],
        "metro": prop["metro"],

        # Valuation outputs
        "list_price": valuation.list_price,
        "noi": valuation.noi,
        "cap_rate": valuation.cap_rate,
        "irr": valuation.irr_projected,
        "equity_multiple": valuation.equity_multiple,

        # Metro fundamentals — None → 0.0 / "unknown" so API never gets null
        "pci_2023": _coerce(market.get("pci_2023"), 0.0),
        "pop_growth": _coerce(market.get("pop_growth"), 0.0),
        "metro_elasticity": _coerce(market.get("metro_elasticity"), 0.0),
        "supply_bucket": market.get("supply_bucket") or "unknown",

        # Risk metrics
        "risk_score": _coerce(market.get("risk_score"), 0.0),
        "risk_bucket": market.get("risk_bucket") or "unknown",

        # Tokenization terms
        "token_symbol": "KJFI01",
        "token_price": initial_price,
        "total_supply": total_supply,
        "lockup_months": 12,

        # NAV fields
        "nav_total": nav_total,
        "nav_per_token": nav_per_token,
        "nav_currency": "USD",

        # AMM / Uniswap guidance
        "amm": "uniswap_v3",
        "quote_asset": "USDC",
        "initial_price": initial_price,
        "seed_token_amount": seed_tokens,
        "seed_quote_amount": seed_quote,
        "fee_tier": 0.003,
        "price_range_low": initial_price * 0.85,
        "price_range_high": initial_price * 1.15,

        # On-chain metadata (populated after deployment)
        "token_address": None,
        "chain_id": None,
        "pool_address": None,
    }

    try:
        r = requests.post(
            f"{API_URL}/properties/",
            json=payload,
            timeout=10,
            headers={"X-API-Key": os.environ.get("KEJAFI_PROD_KEY", "prod_key_placeholder")},
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise Exception(
            f"Cannot connect to API at {API_URL}. Is the FastAPI server running?"
        )
    except requests.exceptions.Timeout:
        raise Exception("API request timed out after 10 seconds.")
    except requests.exceptions.HTTPError as e:
        raise Exception(
            f"API returned error: {e.response.status_code} - {e.response.text}"
        )


# =========================================================
# Stage 1 Integration UI
# =========================================================

def render_stage1_import():
    """UI component to import metro risk data from Stage 1 session state."""
    st.subheader("Import from Stage 1 (Research Engine)")
    
    if not HAS_STAGE1:
        st.warning("Stage 1 (research_engine.py) not found. Run Stage 1 analysis first.")
        return None
    
    # Check for Stage 1 results in session state
    has_stage1_data = (
        "stage1_result_a" in st.session_state or 
        "stage1_result_b" in st.session_state
    )
    
    if not has_stage1_data:
        st.info("No Stage 1 data found. Run Stage 1 analysis first, then return here.")
        return None
    
    # Let user select which metro to import
    options = []
    if "stage1_result_a" in st.session_state:
        metro_a = st.session_state.get("stage1_metro_a_short", "Metro A")
        options.append((metro_a, "a"))
    if "stage1_result_b" in st.session_state:
        metro_b = st.session_state.get("stage1_metro_b_short", "Metro B")
        options.append((metro_b, "b"))
    
    selected = st.selectbox(
        "Select Stage 1 Metro Analysis",
        options,
        format_func=lambda x: f"{x[0]} (from Stage 1 session)",
    )
    
    if st.button("Import Market Data", type="primary"):
        suffix = selected[1]
        result_key = f"stage1_result_{suffix}"
        county_key = f"stage1_county_{suffix}"
        metro_key = f"stage1_metro_{suffix}_short"
        
        result = st.session_state[result_key]
        county = st.session_state.get(county_key)
        metro_short = st.session_state.get(metro_key, selected[0])
        
        # Convert to Stage 2 format
        market = export_metro_for_stage2(metro_short, result, county)
        
        # Store in Stage 2 namespaced session state
        st.session_state["stage2_imported_market"] = market
        st.session_state["stage2_imported_metro"] = metro_short
        
        st.success(f"Imported {metro_short} market data from Stage 1!")
        st.json(market)
        
        return market
    
    # Return previously imported data if available
    return st.session_state.get("stage2_imported_market")


# =========================================================
# Streamlit UI
# =========================================================

st.title("\U0001f3e0 Kejafi Single-Asset Tokenization Platform")

st.markdown(
    "**Stage 2: Select a property in a metro, run valuation with metro fundamentals, "
    "publish to marketplace API.**"
)

# ----------------------------------------------------------
# Step 0: Stage 1 Import (NEW)
# ----------------------------------------------------------
st.header("Step 0: Import Metro Risk Data (Optional)")
imported_market = render_stage1_import()
st.markdown("---")

# ----------------------------------------------------------
# Step 1: Property selection
# ----------------------------------------------------------
st.header("Step 1: Select Property")

col1, col2 = st.columns([1, 2])

with col1:
    selected_key = st.selectbox(
        "Available Properties",
        list(EXAMPLE_PROPERTIES.keys()),
        format_func=lambda x: (
            f"{EXAMPLE_PROPERTIES[x]['address'][:30]}... "
            f"({EXAMPLE_PROPERTIES[x]['property_type']})"
        ),
    )

prop = EXAMPLE_PROPERTIES[selected_key]

with col1:
    st.subheader("Property Details")
    st.write(f"**{prop['address']}**")
    st.write(f"County: {prop['county']}")
    st.write(f"Type: {prop['property_type']}")
    st.write(f"Units: {prop['units']}")
    st.write(f"SQFT: {prop['sqft']:,}")
    st.write(f"Year Built: {prop['year_built']}")
    st.metric("List Price",             f"${prop['list_price']:,.0f}")
    st.metric("Current Monthly Rent",   f"${prop['current_rent']:,.0f}")
    st.metric("Market Monthly Rent",    f"${prop['market_rent']:,.0f}")

with col2:
    st.subheader("Metro fundamentals & risk (from Stage 1)")
    st.info(prop["description"])

    # Use imported Stage 1 data if available
    if imported_market is not None:
        market = imported_market
        st.success(f"Using imported Stage 1 data for {st.session_state.get('stage2_imported_metro', 'selected metro')}")
    else:
        # Fallback to traditional metro profile lookup
        zori_region_name = resolve_metro_label_for_property(prop["metro"])
        st.caption(f"Using ZORI RegionName: **{zori_region_name}**")

        metro_profile = None
        try:
            metro_profile = build_metro_profile(zori_region_name)
        except FileNotFoundError as e:
            st.error(f"Data file not found: {e}")
        except ValueError as e:
            st.error(f"Failed to load metro profile: {e}")
        except Exception as e:
            st.error(f"Unexpected metro profile error: {e}")

        if metro_profile is None:
            st.warning(
                "Metro fundamentals unavailable. "
                "Valuation will run with defaults; cap-rate and growth may be approximate."
            )
            market: Dict = {
                "yoy_rent_growth": 0.03,
                "rent_volatility": 0.08,   # reserved for Stage 3 Monte Carlo
                "trend_label": "stable",
                "pci_2023": None,
                "metro_elasticity": None,
                "supply_bucket": None,
                "pop_growth": None,        # FIX 1: stays None; full_valuation guards it
                "risk_score": 75.0,
                "risk_bucket": "Moderate",
            }
        else:
            market = {
                "yoy_rent_growth":  metro_profile["yoy_rent_growth"],
                "rent_volatility":  metro_profile["rent_volatility"],   # Stage 3 use
                "trend_label":      metro_profile["trend_label"],
                "pci_2023":         metro_profile["pci_2023"],
                "metro_elasticity": metro_profile["metro_elasticity"],
                "supply_bucket":    metro_profile["supply_bucket"],
                "pop_growth":       metro_profile["pop_growth"],
                "risk_score":       metro_profile["risk_score"],
                "risk_bucket":      metro_profile["risk_bucket"],
            }

    c1, c2c, c3 = st.columns(3)
    c1.metric(
        "Rent growth (YoY)",
        f"{market['yoy_rent_growth']*100:.1f}%",
        delta=market["trend_label"],
    )
    c2c.metric("Rent volatility (ann.)", f"{market['rent_volatility']*100:.1f}%")
    c3.metric("Walk Score", prop["walk_score"])

    c4, c5, c6 = st.columns(3)
    c4.metric(
        "Metro PCI 2023",
        f"${market['pci_2023']:,.0f}" if market["pci_2023"] is not None else "n/a",
    )
    if market["metro_elasticity"] is not None:
        c5.metric(
            "Housing elasticity",
            f"{market['metro_elasticity']:.2f}",
            delta=market.get("supply_bucket", ""),
        )
    else:
        c5.metric("Housing elasticity", "n/a")

    c6.metric(
        "Pop growth 13\u201317\u219218\u201322",
        f"{market['pop_growth']*100:.1f}%"
        if market["pop_growth"] is not None
        else "n/a",
    )

    r1, r2 = st.columns(2)
    with r1:
        st.metric(
            "Kejafi metro risk score",
            f"{market['risk_score']:.0f}/100",
            delta=market["risk_bucket"],
        )
    with r2:
        st.caption(
            "Score derived from ZORI-based Monte Carlo (mean, VaR, CVaR) "
            "and metro fundamentals."
        )

# ----------------------------------------------------------
# Step 2: Valuation
# ----------------------------------------------------------
st.header("Step 2: Valuation & Underwriting")

col_v1, col_v2, col_v3 = st.columns(3)
with col_v1:
    leverage = st.slider("Leverage (LTV)", 0.0, 0.80, 0.65, 0.05)
with col_v2:
    interest_rate = st.slider("Interest Rate", 0.04, 0.12, 0.075, 0.005)
with col_v3:
    hold_period = st.slider("Hold Period (years)", 3, 10, 5, 1)

if st.button("\U0001f3af Run Valuation", type="primary"):
    val_engine = PropertyValuation(prop, market)
    valuation = val_engine.full_valuation(leverage, interest_rate, hold_period)

    # FIX 5: namespaced session state keys to avoid cross-page collisions
    st.session_state["stage2_valuation"] = valuation
    st.session_state["stage2_property"] = prop
    st.session_state["stage2_market"] = market

    st.success("Valuation complete (metro assumptions from Stage 1).")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Stabilized NOI",  f"${valuation.noi:,.0f}")
    r2.metric("Value per door",  f"${valuation.value_per_door:,.0f}")
    r3.metric("Cash-on-cash",    f"{valuation.cash_on_cash*100:.1f}%")
    r4.metric("Projected IRR",   f"{valuation.irr_projected*100:.1f}%")

    st.markdown(
        f"**Cap rate drivers:** Supply bucket "
        f"({market.get('supply_bucket', 'n/a')}) and PCI "
        f"${market.get('pci_2023') or 0:,.0f} shift the base cap rate to "
        f"**{valuation.cap_rate*100:.2f}%**."
    )

    st.subheader("Valuation vs. list price")
    fig, ax = plt.subplots(figsize=(10, 4))
    categories = ["List price", "Income value", "Value/door \u00d7 units"]
    values = [
        valuation.list_price / 1000,
        valuation.value_income / 1000,
        valuation.value_per_door * prop["units"] / 1000,
    ]
    colors = ["red" if values[0] > values[1] else "green", "blue", "gray"]
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel("Value ($K)")
    ax.set_title("Valuation comparison")
    for bar, val_ in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"${val_:.0f}K",
            ha="center",
            va="bottom",
        )
    fname = FIG_DIR / f"valuation_comparison_{selected_key}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    st.pyplot(fig)

# ----------------------------------------------------------
# Step 3: Publish to FastAPI
# ----------------------------------------------------------
if "stage2_valuation" in st.session_state:
    st.header("Step 3: Publish to Marketplace API")
    st.markdown(
        "Pushes a unified property object (metro fundamentals + risk, valuation, "
        "token terms) to the FastAPI backend for the investor frontend."
    )

    if st.button("\U0001f4e1 Publish to API"):
        valuation = st.session_state["stage2_valuation"]
        prop      = st.session_state["stage2_property"]
        market    = st.session_state["stage2_market"]
        try:
            res = push_property_to_api(selected_key, valuation, market, prop)
            st.success("Published to API.")
            st.write("Public URL (used by the Next.js frontend):")
            st.code(res.get("public_url", ""), language="text")
            st.json(res)
        except Exception as e:
            st.error(f"Failed to publish: {e}")

st.markdown("---")
st.caption(
    "Kejafi Single-Asset Tokenization | Stage 2 of the Urban Economics & "
    "Tokenized Housing pipeline."
)