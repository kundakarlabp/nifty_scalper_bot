"""Mobile-friendly, read-only Streamlit dashboard for Nifty Scalper Bot.

Deploy this app separately from the trading process (for example on Streamlit
Community Cloud). It reads the bot's existing HTTP health endpoints and never
submits broker orders or mutates trading state.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import requests
import streamlit as st


st.set_page_config(
    page_title="Nifty Scalper Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: .7rem .8rem;
      }
      div[data-testid="stMetricValue"] {font-size: 1.35rem;}
      @media (max-width: 640px) {
        .block-container {padding-left: .7rem; padding-right: .7rem;}
        h1 {font-size: 1.65rem !important;}
        div[data-testid="stMetricValue"] {font-size: 1.1rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _secret(name: str, default: str = "") -> str:
    """Read a value from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or os.getenv(name, default)).strip()


def _base_url() -> str:
    return _secret("BOT_API_URL").rstrip("/")


def _headers() -> dict[str, str]:
    token = _secret("BOT_DASHBOARD_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _get_json(path: str) -> tuple[dict[str, Any] | None, str | None, int | None]:
    base_url = _base_url()
    if not base_url:
        return None, "BOT_API_URL is not configured", None
    try:
        response = requests.get(
            f"{base_url}{path}",
            headers=_headers(),
            timeout=8,
        )
        status_code = response.status_code
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return None, "Unexpected non-object API response", status_code
        return payload, None, status_code
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return None, str(exc), status_code
    except ValueError as exc:
        return None, f"Invalid JSON response: {exc}", None


def _label(value: Any, fallback: str = "—") -> str:
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return str(value)


def _status_banner(livez: dict[str, Any] | None, trading: dict[str, Any] | None) -> None:
    if not livez:
        st.error("🔴 Bot API is unreachable")
        return
    if trading and trading.get("live_orders_armed"):
        st.success("🟢 Bot online — live orders armed")
    elif trading and trading.get("status") == "blocked":
        blocker = trading.get("primary_blocker") or "unknown blocker"
        st.warning(f"🟠 Bot online — trading blocked: {blocker}")
    else:
        st.info("🔵 Bot process online — trading engine starting or not armed")


st.title("📈 Nifty Scalper Monitor")
st.caption("Read-only mobile dashboard • No order controls")

with st.sidebar:
    st.subheader("Connection")
    configured_url = _base_url()
    st.code(configured_url or "BOT_API_URL not set", language=None)
    st.caption("Configure BOT_API_URL in Streamlit app secrets.")
    if st.button("Refresh now", use_container_width=True):
        st.rerun()

livez, livez_error, livez_code = _get_json("/livez")
readyz, readyz_error, readyz_code = _get_json("/readyz")
trading, trading_error, trading_code = _get_json("/health/trading")

_status_banner(livez, trading)

st.caption(f"Last checked: {datetime.now().astimezone().strftime('%d %b %Y, %I:%M:%S %p %Z')}")

col1, col2 = st.columns(2)
with col1:
    st.metric("API", "ONLINE" if livez else "OFFLINE")
with col2:
    st.metric("Bot loaded", _label((livez or {}).get("bot_loaded")))

col3, col4 = st.columns(2)
with col3:
    st.metric("Execution ready", _label((trading or {}).get("ready")))
with col4:
    st.metric("Live orders armed", _label((trading or {}).get("live_orders_armed")))

broker = (trading or {}).get("broker") or {}
reconciliation = (trading or {}).get("reconciliation") or {}

st.subheader("Broker and safety state")
col5, col6 = st.columns(2)
with col5:
    st.metric("Broker ready", _label(broker.get("ready")))
    st.metric("Balance valid", _label(broker.get("balance_valid")))
with col6:
    st.metric("Available balance", _label(broker.get("balance")))
    st.metric("Auth invalid", _label(broker.get("auth_invalid")))

st.subheader("Position reconciliation")
col7, col8 = st.columns(2)
with col7:
    st.metric("Completed", _label(reconciliation.get("completed")))
with col8:
    st.metric("Failed", _label(reconciliation.get("failed")))

unprotected = reconciliation.get("unprotected_positions") or []
if unprotected:
    st.error("Unprotected broker positions detected")
    st.json(unprotected)

blockers = (trading or {}).get("blockers") or (readyz or {}).get("blockers") or []
st.subheader("Current blockers")
if blockers:
    for blocker in blockers:
        st.warning(str(blocker))
else:
    st.success("No reported blockers")

with st.expander("Raw diagnostics"):
    st.write(
        {
            "livez": {"http": livez_code, "data": livez, "error": livez_error},
            "readyz": {"http": readyz_code, "data": readyz, "error": readyz_error},
            "health_trading": {
                "http": trading_code,
                "data": trading,
                "error": trading_error,
            },
        }
    )

st.divider()
st.caption("Open this URL in Chrome on Android, then use Add to Home screen for app-like access.")
