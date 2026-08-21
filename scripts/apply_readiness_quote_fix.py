from pathlib import Path

path = Path("src/nifty_scalper_bot/core/history_readiness.py")
text = path.read_text()
old = '''def _get_cached_quote(ctx: "BotContext", symbol: str) -> Mapping[str, Any]:
    """Return cached quote/tick data without pulling broker APIs."""
    for provider in (getattr(ctx, "data_hub", None), getattr(ctx, "datahub", None)):
        if provider is None:
            continue
        fn = getattr(provider, "get_quote", None)
        if callable(fn):
            try:
                quote = fn(symbol, allow_pull=False)
            except TypeError:
                try:
                    quote = fn(symbol)
                except Exception:
                    quote = None
            except Exception:
                quote = None
            if isinstance(quote, Mapping):
                return quote
    mdm = getattr(ctx, "market_data_manager", None)
    for name in ("get_quote", "get_latest_tick", "get_last_tick"):
        fn = getattr(mdm, name, None)
        if callable(fn):
            try:
                quote = fn(symbol)
            except Exception:
                quote = None
            if isinstance(quote, Mapping):
                return quote
'''
new = '''def _get_cached_quote(ctx: "BotContext", symbol: str) -> Mapping[str, Any]:
    """Return canonical MDM cached tick; use DataHub only as read-facade fallback."""
    mdm = getattr(ctx, "market_data_manager", None)
    for name in ("get_latest_tick", "get_last_tick", "get_quote"):
        fn = getattr(mdm, name, None)
        if callable(fn):
            try:
                quote = fn(symbol)
            except Exception:
                quote = None
            if isinstance(quote, Mapping) and quote:
                return quote
    for provider in (getattr(ctx, "data_hub", None), getattr(ctx, "datahub", None)):
        if provider is None:
            continue
        fn = getattr(provider, "get_quote", None)
        if callable(fn):
            try:
                quote = fn(symbol, allow_pull=False)
            except TypeError:
                try:
                    quote = fn(symbol)
                except Exception:
                    quote = None
            except Exception:
                quote = None
            if isinstance(quote, Mapping):
                return quote
'''
if old not in text:
    raise SystemExit("target readiness helper did not match current branch")
path.write_text(text.replace(old, new, 1))
