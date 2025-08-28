import pandas as pd

from src.strategies.scalping_strategy import EnhancedScalpingStrategy


def make_df(breakout_factor: float) -> pd.DataFrame:
    prices = [100 + i * 0.1 for i in range(60)]
    spot = prices[-1]
    highs = prices.copy()
    highs[-2] = spot * breakout_factor
    data = {
        "open": prices,
        "high": highs,
        "low": prices,
        "close": prices,
        "volume": [1] * 60,
    }
    return pd.DataFrame(data)


def setup_strategy(monkeypatch):
    strat = EnhancedScalpingStrategy(min_bars_for_signal=50)

    # ema21>ema50 and slope>0
    ema21 = pd.Series([i for i in range(60)])
    ema50 = pd.Series([i - 10 for i in range(60)])
    monkeypatch.setattr(strat, "_ema", lambda s, p: ema21 if p == 21 else ema50)

    # vwap below price
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.calculate_vwap",
        lambda df: pd.Series([99] * len(df)),
    )

    # macd line >0 and histogram rising
    macd_line = pd.Series([0.1 * i for i in range(60)])
    macd_hist = pd.Series([0.1 * i for i in range(60)])
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.calculate_macd",
        lambda close: (macd_line, macd_line, macd_hist),
    )

    # rsi within band and rising
    rsi_series = pd.Series([50 + i * 0.2 for i in range(60)])
    monkeypatch.setattr(strat, "_rsi", lambda s, period=14: rsi_series)

    # ATR constant -> pct 0.005
    atr_series = pd.Series([0.5] * 60)
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.compute_atr", lambda df, period=14: atr_series
    )
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.latest_atr_value", lambda s, default=0.0: 0.5
    )

    return strat


def test_trend_playbook_pass(monkeypatch):
    strat = setup_strategy(monkeypatch)
    df = make_df(0.997)  # 0.3% away
    sig = strat.generate_signal(df)
    assert sig is not None


def test_trend_breakout_guard(monkeypatch):
    strat = setup_strategy(monkeypatch)
    df = make_df(0.9995)  # 0.05% away -> block
    sig = strat.generate_signal(df)
    assert sig is None
