import pandas as pd
from src.signals.regime_detector import _pick_col, detect_market_regime, RegimeResult


def test_pick_col_variants():
    df = pd.DataFrame({"adx": [1, 2], "adx_14": [3, 4]})
    col = _pick_col(df, "adx")
    assert col.equals(df["adx"])

    df2 = pd.DataFrame({"adx_14": [1, 2], "adx_7": [3, 4]})
    col2 = _pick_col(df2, "adx")
    assert col2.equals(df2["adx_7"])

    df3 = pd.DataFrame({"foo": [1, 2]})
    assert _pick_col(df3, "adx") is None


def test_detect_market_regime_empty_df():
    res = detect_market_regime(df=pd.DataFrame())
    assert res.regime == "NO_TRADE" and res.reason == "empty_df"


def test_detect_market_regime_handles_missing_close():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    res = detect_market_regime(df=df)
    assert isinstance(res, RegimeResult)


def test_detect_market_regime_computes_bb_width():
    df = pd.DataFrame({"close": list(range(30))})
    res = detect_market_regime(df=df)
    assert isinstance(res, RegimeResult)
