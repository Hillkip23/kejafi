# -*- coding: utf-8 -*-
"""
Kejafi FastAPI backend — v0.4.0 (DeFi Integration Update)

v0.4.0 changes:
- Seeds charlotte_mfk_002 (Property B) for FINE6
- Adds token registration endpoint for post-deployment address updates
- Adds lookup by token symbol for frontend integration
- Returns token_address/pool_address in property responses for Web3 integration
- Added demo data fallback for risk endpoints (no metro_lab_core required)
"""

import os
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.engine.database import Base, engine, get_db
from app.engine.orm import PropertyORM
from app.models.schemas import DeleteResponse, PropertyCreate, PropertyOut, PropertyUpdate

# =========================================================
# App init
# =========================================================

app = FastAPI(title="Kejafi Marketplace API", version="0.4.0")

Base.metadata.create_all(bind=engine)

# =========================================================
# CORS (added localhost:3001 for Next.js dev server alternate)
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8501",
        "http://localhost:8506",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8506",
        "https://kejafi-api.onrender.com",
        "https://kejafi.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# API key auth
# =========================================================

_API_KEY = os.environ.get("KEJAFI_PROD_KEY", "prod_key_placeholder")


def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# =========================================================
# Token Registration Schema
# =========================================================

class TokenRegistration(BaseModel):
    token_address: str
    pool_address: Optional[str] = None
    chain_id: int = 11155111
    token_symbol: Optional[str] = None


# =========================================================
# Demo data for risk endpoints (FALLBACK when metro_lab_core unavailable)
# =========================================================

def _get_demo_series(metro: str):
    """Generate synthetic rent data for demo purposes"""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2015-01-01', periods=120, freq='MS')
    base_rent = 1500
    monthly_growth = 0.0025
    trend = 1 + (np.arange(120) * monthly_growth)
    np.random.seed(42)
    noise = np.random.normal(1, 0.02, 120)
    values = base_rent * trend * noise
    return pd.Series(values, index=dates)


def _get_metro_series(metro: str):
    """Get ZORI rent series for a metro, or return synthetic demo data"""
    try:
        from metro_lab_core import get_series, ZORI_REGION_MAP
        series = get_series(metro)
        if series is None:
            return _get_demo_series(metro)
        return series
    except (ImportError, FileNotFoundError, Exception):
        return _get_demo_series(metro)


def _get_metro_fundamentals(metro: str) -> dict:
    """Get metro fundamentals or return demo data"""
    try:
        from metro_lab_core import get_metro_fundamentals
        return get_metro_fundamentals(metro)
    except Exception:
        return {
            "pci_2023": 75000,
            "pop_growth": 0.08,
            "metro_elasticity": 0.35,
            "supply_bucket": "neutral",
        }


def _sharpe(risk_metrics: dict, risk_free: float = 0.05) -> float:
    er = risk_metrics.get("expected_return", 0)
    vol = risk_metrics.get("volatility", 1e-6)
    return round((er - risk_free) / max(vol, 1e-6), 3)


def _risk_level(var: float) -> str:
    loss = abs(var)
    if loss > 0.25:
        return "HIGH"
    if loss > 0.12:
        return "MODERATE"
    return "LOW"


def _severity_score(results: dict) -> int:
    var = abs(results.get("var_95", 0))
    if var >= 0.30:
        return 90
    if var >= 0.20:
        return 70
    if var >= 0.10:
        return 50
    if var >= 0.05:
        return 30
    return 15


# =========================================================
# ORM helpers
# =========================================================

def _orm_to_out(row: PropertyORM) -> PropertyOut:
    return PropertyOut(
        id=row.id,
        address=row.address,
        county=row.county,
        metro=row.metro,
        list_price=row.list_price,
        noi=row.noi,
        cap_rate=row.cap_rate,
        irr=row.irr,
        equity_multiple=row.equity_multiple,
        pci_2023=row.pci_2023,
        pop_growth=row.pop_growth,
        metro_elasticity=row.metro_elasticity,
        supply_bucket=row.supply_bucket,
        risk_score=row.risk_score,
        risk_bucket=row.risk_bucket,
        token_symbol=row.token_symbol,
        token_price=row.token_price,
        total_supply=row.total_supply,
        lockup_months=row.lockup_months,
        nav_total=row.nav_total,
        nav_per_token=row.nav_per_token,
        nav_currency=row.nav_currency,
        amm=row.amm,
        quote_asset=row.quote_asset,
        initial_price=row.initial_price,
        seed_token_amount=row.seed_token_amount,
        seed_quote_amount=row.seed_quote_amount,
        fee_tier=row.fee_tier,
        price_range_low=row.price_range_low,
        price_range_high=row.price_range_high,
        token_address=row.token_address,
        chain_id=row.chain_id,
        pool_address=row.pool_address,
        is_published=row.is_published,
        public_url=row.public_url,
    )


def _get_or_404(prop_id: str, db: Session) -> PropertyORM:
    row = db.query(PropertyORM).filter(PropertyORM.id == prop_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Property not found")
    return row


# =========================================================
# Startup seed
# =========================================================

@app.on_event("startup")
def seed_db():
    db = next(get_db())
    try:
        if db.query(PropertyORM).filter(PropertyORM.id == "charlotte_mfk_001").first() is None:
            db.add(PropertyORM(
                id="charlotte_mfk_001",
                address="1421 N Caldwell St, Charlotte, NC 28206",
                county="Mecklenburg County, NC",
                metro="Charlotte-Concord-Gastonia, NC-SC",
                list_price=685_000,
                noi=47_808,
                cap_rate=0.0551,
                irr=0.2523,
                equity_multiple=3.08,
                pci_2023=69_588.0,
                pop_growth=0.10,
                metro_elasticity=None,
                supply_bucket=None,
                risk_score=75.0,
                risk_bucket="Moderate",
                token_symbol="FINE5",
                token_price=5.48,
                total_supply=100_000,
                lockup_months=12,
                nav_total=548_000.0,
                nav_per_token=5.48,
                nav_currency="USD",
                amm="uniswap_v3",
                quote_asset="USDC",
                initial_price=5.48,
                seed_token_amount=20_000,
                seed_quote_amount=109_600.0,
                fee_tier=0.003,
                price_range_low=4.66,
                price_range_high=6.30,
                token_address="0x0FB987BEE67FD839cb1158B0712d5e4Be483dd2E",
                chain_id=11155111,
                pool_address="0x0Bf78f76c86153E433dAA5Ac6A88453D30968e27",
                is_published=True,
                public_url="https://app.kejafi.com/properties/charlotte_mfk_001",
            ))
        
        if db.query(PropertyORM).filter(PropertyORM.id == "charlotte_mfk_002").first() is None:
            db.add(PropertyORM(
                id="charlotte_mfk_002",
                address="1423 N Caldwell St, Charlotte, NC 28206",
                county="Mecklenburg County, NC",
                metro="Charlotte-Concord-Gastonia, NC-SC",
                list_price=720_000,
                noi=50_400,
                cap_rate=0.0525,
                irr=0.24,
                equity_multiple=2.95,
                pci_2023=69_588.0,
                pop_growth=0.10,
                metro_elasticity=None,
                supply_bucket=None,
                risk_score=72.0,
                risk_bucket="Moderate",
                token_symbol="FINE6",
                token_price=5.76,
                total_supply=100_000,
                lockup_months=12,
                nav_total=576_000.0,
                nav_per_token=5.76,
                nav_currency="USD",
                amm="uniswap_v3",
                quote_asset="USDC",
                initial_price=5.76,
                seed_token_amount=20_000,
                seed_quote_amount=115_200.0,
                fee_tier=0.003,
                price_range_low=4.90,
                price_range_high=6.62,
                token_address=None,
                chain_id=11155111,
                pool_address=None,
                is_published=True,
                public_url="https://app.kejafi.com/properties/charlotte_mfk_002",
            ))
        
        db.commit()
    finally:
        db.close()


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():
    return {"status": "ok", "service": "kejafi-api", "version": "0.4.0"}


# =========================================================
# Analytics endpoints (with demo data fallback)
# =========================================================

@app.get("/risk/{metro}")
def get_risk(
    metro: str,
    horizon_years: float = Query(default=1.0, ge=0.25, le=10.0),
    confidence: float = Query(default=0.95, ge=0.80, le=0.99),
):
    from app.engine.ou_estimator import OrnsteinUhlenbeckEstimator
    from app.engine.monte_carlo import MonteCarloRiskEngine

    series = _get_metro_series(metro)

    try:
        ou_params = OrnsteinUhlenbeckEstimator().estimate(series)
    except Exception as e:
        raise HTTPException(422, detail=f"OU estimation failed: {e}")

    try:
        risk_metrics = MonteCarloRiskEngine(n_simulations=5000).calculate_var(
            rent_series=series,
            horizon_years=horizon_years,
            confidence=confidence,
            ou_params=ou_params,
        )
    except Exception as e:
        raise HTTPException(422, detail=f"Monte Carlo failed: {e}")

    return {
        "metro": metro,
        "horizon_years": horizon_years,
        "confidence": confidence,
        "ou_parameters": {
            "kappa": ou_params["kappa"],
            "theta": ou_params["theta"],
            "sigma": ou_params["sigma"],
            "half_life": ou_params["half_life"],
            "r_squared": ou_params["r_squared"],
            "long_term_mean": ou_params["theta"],
            "n_observations": ou_params["n_observations"],
        },
        "risk_metrics": risk_metrics,
        "fundamentals": _get_metro_fundamentals(metro),
    }


@app.get("/stress/{metro}")
def get_stress(
    metro: str,
    scenario: str = Query(
        default="BASE_CASE",
        description="BASE_CASE | COVID_SHOCK | GFC_2008 | STAGFLATION",
    ),
):
    from app.engine.ou_estimator import OrnsteinUhlenbeckEstimator
    from app.engine.stress_lab import StressLab

    series = _get_metro_series(metro)

    try:
        ou_params = OrnsteinUhlenbeckEstimator().estimate(series)
    except Exception as e:
        raise HTTPException(422, detail=f"OU estimation failed: {e}")

    lab = StressLab(n_simulations=5000)
    scenario_upper = scenario.upper()

    if scenario_upper not in lab.SCENARIOS:
        raise HTTPException(
            400,
            detail=f"Unknown scenario '{scenario}'. Valid: {list(lab.SCENARIOS.keys())}",
        )

    try:
        stress_results = lab.run_scenario(
            rent_series=series,
            ou_params=ou_params,
            scenario=scenario_upper,
        )
        interpretation = lab.interpret_results(stress_results, scenario_upper)
    except Exception as e:
        raise HTTPException(422, detail=f"Stress simulation failed: {e}")

    return {
        "metro": metro,
        "scenario": scenario_upper,
        "stress_results": stress_results,
        "interpretation": {
            **interpretation,
            "severity_score": _severity_score(stress_results),
            "description": interpretation.get("scenario_description", ""),
            "risk_level": interpretation.get("risk_level", "UNKNOWN"),
        },
    }


@app.get("/cap-rate/{metro}")
def get_cap_rate(metro: str):
    from app.engine.cap_rate import DynamicCapRateModel

    fundamentals = _get_metro_fundamentals(metro)
    elasticity = fundamentals.get("metro_elasticity") or 2.5
    pci = fundamentals.get("pci_2023") or 65_000

    result = DynamicCapRateModel().calculate(
        metro=metro, elasticity=elasticity, pci=pci,
    )

    rent_growth_forecast = None
    try:
        series = _get_metro_series(metro)
        if len(series) > 12:
            rent_growth_forecast = float(series.iloc[-1] / series.iloc[-13] - 1)
    except Exception:
        pass

    return {
        **result,
        "metro": metro,
        "rent_growth_forecast": rent_growth_forecast,
        "pop_growth": fundamentals.get("pop_growth"),
        "pci": pci,
    }


@app.get("/compare")
def compare_metros(
    metros: List[str] = Query(..., description="?metros=Charlotte&metros=Atlanta"),
    horizon_years: float = Query(default=1.0, ge=0.25, le=10.0),
    confidence: float = Query(default=0.95, ge=0.80, le=0.99),
):
    from app.engine.ou_estimator import OrnsteinUhlenbeckEstimator
    from app.engine.monte_carlo import MonteCarloRiskEngine
    from app.engine.cap_rate import DynamicCapRateModel

    ou_estimator = OrnsteinUhlenbeckEstimator()
    mc = MonteCarloRiskEngine(n_simulations=2000)
    cap_model = DynamicCapRateModel()

    comparison, errors = [], []

    for metro in metros:
        try:
            series = _get_metro_series(metro)
            ou_params = ou_estimator.estimate(series)
            risk_metrics = mc.calculate_var(
                rent_series=series,
                horizon_years=horizon_years,
                confidence=confidence,
                ou_params=ou_params,
            )
            fundamentals = _get_metro_fundamentals(metro)
            elasticity = fundamentals.get("metro_elasticity") or 2.5
            pci = fundamentals.get("pci_2023") or 65_000
            cap_result = cap_model.calculate(metro=metro, elasticity=elasticity, pci=pci)

            var_key = f"var_{int(confidence * 100)}"
            cvar_key = f"cvar_{int(confidence * 100)}"

            comparison.append({
                "metro": metro,
                "expected_return": risk_metrics.get("expected_return", 0),
                "volatility": risk_metrics.get("volatility", 0),
                "var_95": risk_metrics.get(var_key, risk_metrics.get("var_95", 0)),
                "cvar_95": risk_metrics.get(cvar_key, risk_metrics.get("cvar_95", 0)),
                "sharpe_ratio": _sharpe(risk_metrics),
                "cap_rate": cap_result.get("final_cap_rate"),
                "kappa": ou_params.get("kappa"),
                "half_life": ou_params.get("half_life"),
                "pop_growth": fundamentals.get("pop_growth"),
                "pci_2023": pci,
                "risk_level": _risk_level(
                    risk_metrics.get(var_key, risk_metrics.get("var_95", 0))
                ),
            })

        except HTTPException as e:
            errors.append({"metro": metro, "error": e.detail})
        except Exception as e:
            errors.append({"metro": metro, "error": str(e)})

    return {
        "comparison": comparison,
        "errors": errors,
        "horizon_years": horizon_years,
        "confidence": confidence,
    }


# =========================================================
# Properties — read (public)
# =========================================================

@app.get("/properties/", response_model=List[PropertyOut])
def list_properties(db: Session = Depends(get_db)):
    return [_orm_to_out(r) for r in db.query(PropertyORM).all()]


@app.get("/properties/{prop_id}", response_model=PropertyOut)
def get_property(prop_id: str, db: Session = Depends(get_db)):
    return _orm_to_out(_get_or_404(prop_id, db))


@app.get("/tokens/", response_model=List[PropertyOut])
def list_tokens(db: Session = Depends(get_db)):
    rows = db.query(PropertyORM).filter(PropertyORM.is_published == True).all()
    return [_orm_to_out(r) for r in rows]


@app.get("/tokens/by-symbol/{symbol}", response_model=PropertyOut)
def get_token_by_symbol(symbol: str, db: Session = Depends(get_db)):
    row = (
        db.query(PropertyORM)
        .filter(PropertyORM.token_symbol == symbol.upper())
        .filter(PropertyORM.is_published == True)
        .first()
    )
    if not row:
        raise HTTPException(
            status_code=404, 
            detail=f"Token symbol {symbol} not found"
        )
    return _orm_to_out(row)


# =========================================================
# Token registration (API key required)
# =========================================================

@app.post(
    "/properties/{prop_id}/tokens",
    response_model=PropertyOut,
    dependencies=[Depends(require_api_key)],
)
def register_tokens(
    prop_id: str,
    registration: TokenRegistration,
    db: Session = Depends(get_db),
):
    row = _get_or_404(prop_id, db)
    
    row.token_address = registration.token_address
    row.pool_address = registration.pool_address
    row.chain_id = registration.chain_id
    if registration.token_symbol:
        row.token_symbol = registration.token_symbol
    
    db.commit()
    db.refresh(row)
    
    return _orm_to_out(row)


@app.patch(
    "/properties/{prop_id}/tokens",
    response_model=PropertyOut,
    dependencies=[Depends(require_api_key)],
)
def update_token_addresses(
    prop_id: str,
    registration: TokenRegistration,
    db: Session = Depends(get_db),
):
    return register_tokens(prop_id, registration, db)


# =========================================================
# Properties — write (API key required)
# =========================================================

@app.post(
    "/properties/",
    response_model=PropertyOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def create_property(prop: PropertyCreate, db: Session = Depends(get_db)):
    existing = db.query(PropertyORM).filter(PropertyORM.id == prop.id).first()
    if existing:
        db.delete(existing)
        db.flush()
    row = PropertyORM(
        **prop.model_dump(),
        is_published=True,
        public_url=f"https://app.kejafi.com/properties/{prop.id}",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _orm_to_out(row)


@app.put(
    "/properties/{prop_id}",
    response_model=PropertyOut,
    dependencies=[Depends(require_api_key)],
)
def update_property(prop_id: str, updates: PropertyUpdate, db: Session = Depends(get_db)):
    row = _get_or_404(prop_id, db)
    for field, value in updates.model_dump(exclude_unset=True).items():
        setattr(row, field, value)
    db.commit()
    db.refresh(row)
    return _orm_to_out(row)


@app.delete(
    "/properties/{prop_id}",
    response_model=DeleteResponse,
    dependencies=[Depends(require_api_key)],
)
def delete_property(prop_id: str, db: Session = Depends(get_db)):
    row = _get_or_404(prop_id, db)
    db.delete(row)
    db.commit()
    return DeleteResponse(id=prop_id)


# =========================================================
# Swaps
# =========================================================

class SwapIntent(BaseModel):
    property_id: str
    pay_token: str
    pay_amount: float
    receive_token: str
    receive_amount: float


_swap_intents: List[SwapIntent] = []


@app.post("/swaps", dependencies=[Depends(require_api_key)])
def create_swap(intent: SwapIntent, db: Session = Depends(get_db)):
    _get_or_404(intent.property_id, db)
    _swap_intents.append(intent)
    return {"status": "ok", "queue_position": len(_swap_intents)}


@app.get("/swaps", response_model=List[SwapIntent])
def list_swaps():
    return _swap_intents