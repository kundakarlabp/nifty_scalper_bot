from pydantic_settings import BaseSettings
from pydantic import Field


class AppSettings(BaseSettings):
    # --- Modes / Logging ---
    enable_live_trading: bool = Field(default=False, env="ENABLE_LIVE_TRADING")
    allow_offhours_testing: bool = Field(default=False, env="ALLOW_OFFHOURS_TESTING")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # --- Scheduling / Hours ---
    time_filter_start: str = Field(default="09:20", env="TIME_FILTER_START")
    time_filter_end: str = Field(default="15:20", env="TIME_FILTER_END")
    data_lookback_minutes: int = Field(default=15, env="DATA__LOOKBACK_MINUTES")

    # --- Instruments ---
    instruments_spot_symbol: str = Field(default="NSE:NIFTY 50", env="INSTRUMENTS__SPOT_SYMBOL")
    instruments_trade_symbol: str = Field(default="NIFTY", env="INSTRUMENTS__TRADE_SYMBOL")
    instruments_trade_exchange: str = Field(default="NFO", env="INSTRUMENTS__TRADE_EXCHANGE")
    instruments_instrument_token: int = Field(default=256265, env="INSTRUMENTS__INSTRUMENT_TOKEN")
    instruments_nifty_lot_size: int = Field(default=75, env="INSTRUMENTS__NIFTY_LOT_SIZE")
    instruments_strike_range: int = Field(default=3, env="INSTRUMENTS__STRIKE_RANGE")
    instruments_min_lots: int = Field(default=1, env="INSTRUMENTS__MIN_LOTS")
    instruments_max_lots: int = Field(default=15, env="INSTRUMENTS__MAX_LOTS")

    # --- Strategy Core ---
    strategy_min_signal_score: int = Field(default=2, env="STRATEGY__MIN_SIGNAL_SCORE")
    strategy_confidence_threshold: float = Field(default=2.0, env="STRATEGY__CONFIDENCE_THRESHOLD")
    strategy_atr_period: int = Field(default=14, env="STRATEGY__ATR_PERIOD")
    strategy_atr_sl_multiplier: float = Field(default=1.5, env="STRATEGY__ATR_SL_MULTIPLIER")
    strategy_atr_tp_multiplier: float = Field(default=3.0, env="STRATEGY__ATR_TP_MULTIPLIER")
    strategy_sl_confidence_adj: float = Field(default=0.12, env="STRATEGY__SL_CONFIDENCE_ADJ")
    strategy_tp_confidence_adj: float = Field(default=0.35, env="STRATEGY__TP_CONFIDENCE_ADJ")
    strategy_min_bars_for_signal: int = Field(default=10, env="STRATEGY__MIN_BARS_FOR_SIGNAL")

    # --- Risk / Limits ---
    risk_default_equity: float = Field(default=30000, env="RISK__DEFAULT_EQUITY")
    risk_per_trade: float = Field(default=0.025, env="RISK__RISK_PER_TRADE")
    risk_max_trades_per_day: int = Field(default=30, env="RISK__MAX_TRADES_PER_DAY")
    risk_consecutive_loss_limit: int = Field(default=3, env="RISK__CONSECUTIVE_LOSS_LIMIT")
    risk_max_daily_drawdown_pct: float = Field(default=0.05, env="RISK__MAX_DAILY_DRAWDOWN_PCT")

    # --- Execution ---
    executor_exchange: str = Field(default="NFO", env="EXECUTOR__EXCHANGE")
    executor_order_product: str = Field(default="NRML", env="EXECUTOR__ORDER_PRODUCT")
    executor_order_variety: str = Field(default="regular", env="EXECUTOR__ORDER_VARIETY")
    executor_entry_order_type: str = Field(default="LIMIT", env="EXECUTOR__ENTRY_ORDER_TYPE")
    executor_tick_size: float = Field(default=0.05, env="EXECUTOR__TICK_SIZE")
    executor_exchange_freeze_qty: int = Field(default=1800, env="EXECUTOR__EXCHANGE_FREEZE_QTY")
    executor_preferred_exit_mode: str = Field(default="REGULAR", env="EXECUTOR__PREFERRED_EXIT_MODE")
    executor_use_slm_exit: bool = Field(default=True, env="EXECUTOR__USE_SLM_EXIT")
    executor_partial_tp_enable: bool = Field(default=True, env="EXECUTOR__PARTIAL_TP_ENABLE")
    executor_tp1_qty_ratio: float = Field(default=0.5, env="EXECUTOR__TP1_QTY_RATIO")
    executor_breakeven_ticks: int = Field(default=2, env="EXECUTOR__BREAKEVEN_TICKS")
    executor_enable_trailing: bool = Field(default=True, env="EXECUTOR__ENABLE_TRAILING")
    executor_trailing_atr_multiplier: float = Field(default=1.5, env="EXECUTOR__TRAILING_ATR_MULTIPLIER")
    executor_fee_per_lot: float = Field(default=20.0, env="EXECUTOR__FEE_PER_LOT")

    # --- Telegram ---
    telegram_enabled: bool = Field(default=False, env="TELEGRAM_ENABLED")
    telegram_bot_token: str = Field(default="", env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", env="TELEGRAM_CHAT_ID")

    # --- Zerodha / Broker Keys ---
    zerodha_api_key: str = Field(default="", env="ZERODHA_API_KEY")
    zerodha_api_secret: str = Field(default="", env="ZERODHA_API_SECRET")
    zerodha_access_token: str = Field(default="", env="ZERODHA_ACCESS_TOKEN")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = AppSettings()