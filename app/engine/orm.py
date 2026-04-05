# -*- coding: utf-8 -*-
"""
SQLAlchemy ORM model — maps the PropertyOut schema to a SQLite table.
Every field is nullable except the primary key so that partial publishes
and future schema additions don't break existing rows.
"""

from sqlalchemy import Boolean, Column, Float, Integer, String
from app.engine.database import Base


class PropertyORM(Base):
    __tablename__ = "properties"

    id               = Column(String, primary_key=True, index=True)

    # Core info
    address          = Column(String, nullable=False)
    county           = Column(String, nullable=False)
    metro            = Column(String, nullable=False)

    # Valuation
    list_price       = Column(Float, nullable=False)
    noi              = Column(Float, nullable=False)
    cap_rate         = Column(Float, nullable=False)
    irr              = Column(Float, nullable=False)
    equity_multiple  = Column(Float, nullable=False)

    # Metro fundamentals
    pci_2023         = Column(Float, nullable=True)
    pop_growth       = Column(Float, nullable=True)
    metro_elasticity = Column(Float, nullable=True)
    supply_bucket    = Column(String, nullable=True)

    # Risk
    risk_score       = Column(Float, nullable=True)
    risk_bucket      = Column(String, nullable=True)

    # Tokenization terms
    token_symbol     = Column(String, nullable=False)
    token_price      = Column(Float, nullable=False)
    total_supply     = Column(Integer, nullable=False)
    lockup_months    = Column(Integer, nullable=False)

    # NAV
    nav_total        = Column(Float, nullable=True)
    nav_per_token    = Column(Float, nullable=True)
    nav_currency     = Column(String, nullable=True, default="USD")

    # AMM
    amm              = Column(String, nullable=True)
    quote_asset      = Column(String, nullable=True)
    initial_price    = Column(Float, nullable=True)
    seed_token_amount = Column(Integer, nullable=True)
    seed_quote_amount = Column(Float, nullable=True)
    fee_tier         = Column(Float, nullable=True)
    price_range_low  = Column(Float, nullable=True)
    price_range_high = Column(Float, nullable=True)

    # On-chain metadata
    token_address    = Column(String, nullable=True)
    chain_id         = Column(Integer, nullable=True)
    pool_address     = Column(String, nullable=True)

    # Marketplace status
    is_published     = Column(Boolean, default=False, nullable=False)
    public_url       = Column(String, nullable=False)