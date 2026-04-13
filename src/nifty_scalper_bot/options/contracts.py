"""Single source of truth for NIFTY options contracts.

This module provides efficient gathering and distribution of options contract
data to runners, strategies, and other components. It uses InstrumentManager
as the underlying data source, eliminating the need for InstrumentResolver.

Usage::

    contracts = OptionsContractStore(instrument_manager)
    contracts.load()
    
    # Get current trading universe
    symbols = contracts.get_trading_universe(spot_price=25600.0)
    
    # Get token for a symbol
    token = contracts.get_token("NIFTY26APR25600CE")
    
    # Get symbol for a token
    symbol = contracts.get_symbol(14074882)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set

LOGGER = logging.getLogger("nifty_scalper_bot.options.contracts")


@dataclass(slots=True)
class OptionContract:
    """Represents a single option contract with all necessary metadata."""
    
    instrument_token: int
    tradingsymbol: str
    expiry: date
    strike: float
    instrument_type: str  # "CE" or "PE"
    lot_size: int = 75
    exchange: str = "NFO"
    
    @property
    def symbol_key(self) -> str:
        """Return the symbol key used for lookups."""
        return self.tradingsymbol.upper()
    
    @property
    def qualified_symbol(self) -> str:
        """Return exchange-qualified symbol (e.g., 'NFO:NIFTY26APR25600CE')."""
        return f"{self.exchange}:{self.tradingsymbol}"


@dataclass
class ContractsSnapshot:
    """Snapshot of all contracts at a point in time."""
    
    timestamp: datetime
    spot_price: float
    atm_strike: int
    active_expiry: date
    ce_contracts: List[OptionContract] = field(default_factory=list)
    pe_contracts: List[OptionContract] = field(default_factory=list)
    future_token: Optional[int] = None
    spot_token: Optional[int] = None


class OptionsContractStore:
    """Central repository for NIFTY options contracts.
    
    This class serves as the single source of truth for all options-related
    data, providing efficient access to contracts, tokens, and symbols.
    """
    
    def __init__(self, instrument_manager: Any) -> None:
        """Initialize the contract store.
        
        Args:
            instrument_manager: InstrumentManager instance for token/symbol lookups.
        
        Raises:
            ValueError: If instrument_manager is None.
        """
        if instrument_manager is None:
            raise ValueError("OptionsContractStore requires a valid InstrumentManager")
        
        self._im = instrument_manager
        self._contracts: Dict[str, OptionContract] = {}  # symbol_key -> contract
        self._tokens: Dict[int, str] = {}  # token -> symbol_key
        self._by_expiry: Dict[date, List[OptionContract]] = {}
        self._by_strike: Dict[float, List[OptionContract]] = {}
        self._futures: Dict[date, OptionContract] = {}  # expiry -> future contract
        self._spot_token: Optional[int] = None
        self._loaded = False
        self._last_update: Optional[datetime] = None
    
    def load(self) -> None:
        """Load all NIFTY option contracts from InstrumentManager.
        
        This method populates internal caches with all available NIFTY options,
        futures, and spot instruments.
        
        Raises:
            RuntimeError: If InstrumentManager is not loaded.
        """
        if not self._im.is_loaded():
            raise RuntimeError("InstrumentManager must be loaded before loading contracts")
        
        LOGGER.info("Loading NIFTY options contracts from InstrumentManager...")
        
        # Clear existing caches
        self._contracts.clear()
        self._tokens.clear()
        self._by_expiry.clear()
        self._by_strike.clear()
        self._futures.clear()
        
        # Load spot token
        self._spot_token = 256265  # NIFTY spot well-known token
        
        # Get all NIFTY option contracts from InstrumentManager
        option_contracts = self._im.get_option_contracts("NIFTY")
        
        count = 0
        for contract_data in option_contracts:
            try:
                contract = OptionContract(
                    instrument_token=int(contract_data["instrument_token"]),
                    tradingsymbol=str(contract_data["tradingsymbol"]),
                    expiry=contract_data["expiry"],
                    strike=float(contract_data["strike"]),
                    instrument_type=str(contract_data["instrument_type"]),
                    lot_size=int(contract_data.get("lot_size", 75)),
                    exchange="NFO",
                )
                
                # Index by symbol
                self._contracts[contract.symbol_key] = contract
                self._tokens[contract.instrument_token] = contract.symbol_key
                
                # Index by expiry
                self._by_expiry.setdefault(contract.expiry, []).append(contract)
                
                # Index by strike
                self._by_strike.setdefault(contract.strike, []).append(contract)
                
                count += 1
                
            except (KeyError, TypeError, ValueError) as e:
                LOGGER.debug("Skipping malformed contract data: %s", e)
        
        # Load futures
        futures_list = self._im.get_future_contracts("NIFTY") if hasattr(self._im, "get_future_contracts") else []
        for fut_data in futures_list:
            try:
                expiry = fut_data.get("expiry")
                if expiry:
                    fut_contract = OptionContract(
                        instrument_token=int(fut_data["instrument_token"]),
                        tradingsymbol=str(fut_data["tradingsymbol"]),
                        expiry=expiry,
                        strike=0.0,
                        instrument_type="FUT",
                        lot_size=int(fut_data.get("lot_size", 75)),
                        exchange="NFO",
                    )
                    self._futures[expiry] = fut_contract
            except (KeyError, TypeError, ValueError):
                pass
        
        self._loaded = True
        self._last_update = datetime.now()
        
        LOGGER.info(
            "Loaded %d NIFTY option contracts, %d expiries, %d futures",
            count,
            len(self._by_expiry),
            len(self._futures),
        )
    
    def get_token(self, symbol: str) -> int:
        """Get instrument token for a symbol.
        
        Args:
            symbol: Trading symbol (with or without exchange prefix).
        
        Returns:
            Instrument token as integer.
        
        Raises:
            RuntimeError: If symbol cannot be resolved.
        """
        # Try direct lookup first
        key = str(symbol).strip().upper()
        
        # Remove exchange prefix if present
        if ":" in key:
            bare_symbol = key.split(":", 1)[1]
        else:
            bare_symbol = key
        
        # Check our cache
        if bare_symbol in self._contracts:
            return self._contracts[bare_symbol].instrument_token
        
        # Fall back to InstrumentManager
        try:
            return self._im.get_token(symbol)
        except RuntimeError:
            raise RuntimeError(f"Cannot resolve token for symbol: {symbol}")
    
    def get_symbol(self, token: int) -> Optional[str]:
        """Get trading symbol for a token.
        
        Args:
            token: Instrument token.
        
        Returns:
            Trading symbol string or None if not found.
        """
        token = int(token)
        
        # Check our cache first
        if token in self._tokens:
            return self._tokens[token]
        
        # Fall back to InstrumentManager
        return self._im.get_symbol(token)
    
    def get_exchange(self, token: int) -> str:
        """Get exchange for a token.
        
        Args:
            token: Instrument token.
        
        Returns:
            Exchange string (default "NFO").
        """
        return self._im.get_exchange(int(token))
    
    def get_trading_universe(
        self,
        spot_price: float,
        strikes_around_atm: int = 2,
        strike_step: int = 50,
        include_futures: bool = True,
        include_spot: bool = True,
    ) -> List[str]:
        """Get the current trading universe of option symbols.
        
        Args:
            spot_price: Current NIFTY spot price for ATM calculation.
            strikes_around_atm: Number of strikes around ATM to include.
            strike_step: Strike interval (default 50 for NIFTY).
            include_futures: Whether to include futures symbol.
            include_spot: Whether to include spot symbol.
        
        Returns:
            List of trading symbols for the universe.
        """
        if spot_price <= 0:
            raise ValueError("spot_price must be positive")
        
        # Calculate ATM strike
        atm_strike = int(round(spot_price / strike_step) * strike_step)
        
        # Find nearest expiry
        today = date.today()
        future_expiries = sorted([exp for exp in self._by_expiry.keys() if exp >= today])
        
        if not future_expiries:
            raise RuntimeError("No valid expiry dates found")
        
        active_expiry = future_expiries[0]
        
        # Generate target strikes
        target_strikes = {
            atm_strike + (i * strike_step)
            for i in range(-strikes_around_atm, strikes_around_atm + 1)
        }
        
        # Collect symbols
        symbols: List[str] = []
        
        # Add spot
        if include_spot and self._spot_token:
            symbols.append("NSE:NIFTY 50")
        
        # Add futures
        if include_futures and active_expiry in self._futures:
            symbols.append(self._futures[active_expiry].qualified_symbol)
        
        # Add options
        contracts_for_expiry = self._by_expiry.get(active_expiry, [])
        for contract in contracts_for_expiry:
            if contract.strike in target_strikes:
                symbols.append(contract.qualified_symbol)
        
        LOGGER.info(
            "Trading universe: spot=%.2f ATM=%d expiry=%s symbols=%d",
            spot_price,
            atm_strike,
            active_expiry,
            len(symbols),
        )
        
        return symbols
    
    def get_snapshot(self, spot_price: float) -> ContractsSnapshot:
        """Get a complete snapshot of current contracts.
        
        Args:
            spot_price: Current spot price.
        
        Returns:
            ContractsSnapshot with all current contract data.
        """
        atm_strike = int(round(spot_price / 50) * 50)
        today = date.today()
        future_expiries = sorted([exp for exp in self._by_expiry.keys() if exp >= today])
        active_expiry = future_expiries[0] if future_expiries else today
        
        contracts_for_expiry = self._by_expiry.get(active_expiry, [])
        ce_contracts = [c for c in contracts_for_expiry if c.instrument_type == "CE"]
        pe_contracts = [c for c in contracts_for_expiry if c.instrument_type == "PE"]
        
        future_token = None
        if active_expiry in self._futures:
            future_token = self._futures[active_expiry].instrument_token
        
        return ContractsSnapshot(
            timestamp=datetime.now(),
            spot_price=spot_price,
            atm_strike=atm_strike,
            active_expiry=active_expiry,
            ce_contracts=ce_contracts,
            pe_contracts=pe_contracts,
            future_token=future_token,
            spot_token=self._spot_token,
        )
    
    def is_loaded(self) -> bool:
        """Check if contracts have been loaded."""
        return self._loaded
    
    def last_update(self) -> Optional[datetime]:
        """Get timestamp of last update."""
        return self._last_update
    
    def contract_count(self) -> int:
        """Get total number of loaded contracts."""
        return len(self._contracts)
