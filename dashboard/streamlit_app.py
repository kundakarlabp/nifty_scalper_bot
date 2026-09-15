"""Read-only observability console for Nifty Scalper Bot."""
from __future__ import annotations

import csv
import io
import os
import re
from collections import Counter
from datetime import date, datetime, time
from typing import Any

import requests
import streamlit as st

st.set_page_config(page_title="Nifty Scalper Observability", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1250px}div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:.65rem .75rem}div[data-testid="stMetricValue"]{font-size:1.25rem}@media(max-width:640px){.block-container{padding-left:.6rem;padding-right:.6rem}h1{font-size:1.55rem!important}}</style>""", unsafe_allow_html=True)

LOG_RE = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) IST\s+(?P<message>.*)$")
FUNNEL = {
    "Evaluations": ("RUNNER", "evaluation"),
    "Signals": ("SIGNAL_GENERATED", "signal generated"),
    "Candidates": ("CANDIDATE", "candidate generated"),
    "Approved": ("APPROVED_CANDIDATE", "approved candidate"),
    "Orders": ("ORDER_SENT", "Sending Order", "TRADE_ATTEMPT"),
    "Fills": ("FILLED", "ORDER_COMPLETE"),
    "Closed": ("TRADE_CLOSED", "EXIT"),
}
BLOCK_TERMS = ("BLOCK", "REJECT", "SKIP", "DENIED", "SUPPRESS", "NOT_READY", "UNAVAILABLE")
IMPORTANT_TERMS = ("ERROR", "EXCEPTION", "RESTART", "WEBSOCKET", "DISCONNECT", "RECONCIL", "ORDER_", "TRADE_", "SIGNAL_", "CANDIDATE", "BLOCK")


def secret(name: str, default: str = "") -> str:
    try: value = st.secrets.get(name, default)
    except Exception: value = default
    return str(value or os.getenv(name, default)).strip()


def base_url() -> str: return secret("BOT_API_URL").rstrip("/")
def headers() -> dict[str, str]:
    token = secret("BOT_DASHBOARD_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_json(path: str) -> tuple[dict[str, Any] | None, str | None, int | None]:
    if not base_url(): return None, "BOT_API_URL is not configured", None
    try:
        r = requests.get(f"{base_url()}{path}", headers=headers(), timeout=8)
        code = r.status_code; r.raise_for_status(); data = r.json()
        return (data if isinstance(data, dict) else None), (None if isinstance(data, dict) else "Unexpected API response"), code
    except Exception as exc:
        return None, str(exc), getattr(getattr(exc, "response", None), "status_code", None)


def fetch_logs(lines: int = 50000) -> tuple[list[tuple[datetime, str]], str | None, bool]:
    """Fetch a large recent window. coverage_complete is conservative, never calls a truncated day complete."""
    if not base_url(): return [], "BOT_API_URL is not configured", False
    try:
        r = requests.get(f"{base_url()}/admin/logs/download", params={"fmt":"txt", "lines":lines}, headers=headers(), timeout=30)
        r.raise_for_status()
    except Exception as exc: return [], f"Log download failed: {exc}", False
    parsed=[]
    for raw in r.text.splitlines():
        m=LOG_RE.match(raw.strip())
        if not m: continue
        try: stamp=datetime.strptime(m.group("timestamp"), "%Y-%m-%d %H:%M:%S")
        except ValueError: continue
        parsed.append((stamp,m.group("message")))
    # If response approaches requested cap, assume truncation. Otherwise coverage still depends on first timestamp.
    return parsed, None, len(r.text.splitlines()) < lines


def day_rows(rows: list[tuple[datetime,str]], d: date) -> list[tuple[datetime,str]]:
    lo=datetime.combine(d,time(9,15)); hi=datetime.combine(d,time(15,30))
    return [(ts,msg) for ts,msg in rows if lo <= ts <= hi]


def funnel_counts(rows: list[tuple[datetime,str]]) -> dict[str,int]:
    out={}
    for label,terms in FUNNEL.items(): out[label]=sum(1 for _,m in rows if any(t.lower() in m.lower() for t in terms))
    return out


def blocker_counts(rows: list[tuple[datetime,str]]) -> Counter[str]:
    c=Counter()
    for _,msg in rows:
        up=msg.upper()
        if not any(t in up for t in BLOCK_TERMS): continue
        # Prefer explicit reason-like tokens while keeping parser generic across strategy implementations.
        reason="other"
        for pattern in (r"(?:reason|blocker|gate)[=: ]+([A-Za-z0-9_.-]+)", r"([A-Z][A-Z0-9_]{3,}(?:BLOCK|REJECT|SKIP)[A-Z0-9_]*)"):
            m=re.search(pattern,msg,re.I)
            if m: reason=m.group(1).lower(); break
        c[reason]+=1
    return c


def csv_bytes(rows:list[tuple[datetime,str]]) -> bytes:
    buf=io.StringIO(); w=csv.writer(buf); w.writerow(["timestamp_ist","message"])
    w.writerows([(f"{ts:%Y-%m-%d %H:%M:%S} IST",m) for ts,m in rows])
    return buf.getvalue().encode("utf-8-sig")


def label(v:Any)->str:
    if v is None or v=="": return "—"
    if isinstance(v,bool): return "YES" if v else "NO"
    return str(v)

st.title("📈 Nifty Scalper Observability")
st.caption("Read-only • execution path is never dependent on this dashboard")
with st.sidebar:
    st.subheader("Connection"); st.code(base_url() or "BOT_API_URL not set", language=None)
    log_limit=st.selectbox("Diagnostic log window", [20000,50000], index=1)
    if st.button("Refresh now", use_container_width=True): st.rerun()

livez,live_err,live_code=get_json("/livez"); readyz,ready_err,ready_code=get_json("/readyz"); trading,trading_err,trading_code=get_json("/health/trading")
if not livez: st.error("Bot API unreachable")
elif trading and trading.get("live_orders_armed"): st.success("Bot online — LIVE orders armed")
elif trading and trading.get("status")=="blocked": st.warning(f"Bot online — blocked: {trading.get('primary_blocker') or 'unknown'}")
else: st.info("Bot process online — not currently armed")

cols=st.columns(4)
cols[0].metric("API", "ONLINE" if livez else "OFFLINE"); cols[1].metric("Bot loaded",label((livez or {}).get("bot_loaded")))
cols[2].metric("Execution ready",label((trading or {}).get("ready"))); cols[3].metric("Live armed",label((trading or {}).get("live_orders_armed")))

rows,log_error,below_cap=fetch_logs(log_limit)
selected=st.date_input("Trading date",value=date.today(),max_value=date.today())
today_rows=day_rows(rows,selected)
first_ts=today_rows[0][0] if today_rows else None
coverage_complete=bool(today_rows and below_cap and first_ts.time() <= time(9,16))

live_tab,funnel_tab,timeline_tab,logs_tab,diag_tab=st.tabs(["Live","Trading funnel","Timeline","Market logs","Diagnostics"])
with live_tab:
    broker=(trading or {}).get("broker") or {}; recon=(trading or {}).get("reconciliation") or {}
    c=st.columns(4); c[0].metric("Broker ready",label(broker.get("ready"))); c[1].metric("Balance",label(broker.get("balance"))); c[2].metric("Reconciled",label(recon.get("completed"))); c[3].metric("Recon failed",label(recon.get("failed")))
    blockers=(trading or {}).get("blockers") or (readyz or {}).get("blockers") or []
    if blockers:
        st.subheader("Current blockers")
        for b in blockers: st.warning(str(b))
    else: st.success("No current blockers reported")
    unprotected=recon.get("unprotected_positions") or []
    if unprotected: st.error("Unprotected broker positions detected"); st.json(unprotected)

with funnel_tab:
    st.caption(f"Observed market-hours events for {selected.isoformat()}. Counts are log-derived diagnostics, not broker books.")
    counts=funnel_counts(today_rows); fc=st.columns(len(counts))
    for col,(name,value) in zip(fc,counts.items()): col.metric(name,f"{value:,}")
    bc=blocker_counts(today_rows)
    st.subheader("Suppression / rejection reasons")
    if bc:
        st.dataframe([{"reason":k,"events":v} for k,v in bc.most_common(20)],use_container_width=True,hide_index=True)
    else: st.info("No blocker/rejection events found in the available window.")

with timeline_tab:
    significant=[(ts,msg) for ts,msg in today_rows if any(t in msg.upper() for t in IMPORTANT_TERMS)]
    st.caption(f"{len(significant):,} significant events")
    for ts,msg in significant[-500:][::-1]: st.text(f"{ts:%H:%M:%S}  {msg}")

with logs_tab:
    if log_error: st.error(log_error)
    if today_rows:
        if coverage_complete: st.success(f"Market-day coverage appears complete: {len(today_rows):,} timestamped rows.")
        else: st.warning(f"Partial/uncertain coverage: {len(today_rows):,} rows. This view will not label a bounded recent-log window as a complete trading day.")
        query=st.text_input("Filter logs",placeholder="RUNNER, ORDER, blocker, symbol...")
        shown=[r for r in today_rows if not query or query.lower() in r[1].lower()]
        st.text_area("Log",value="\n".join(f"{ts:%H:%M:%S} {m}" for ts,m in shown[-5000:]),height=500)
        st.download_button("Download available market-hours CSV",csv_bytes(today_rows),file_name=f"niftybot-market-logs-{selected.isoformat()}.csv",mime="text/csv",use_container_width=True)
    else: st.info("No timestamped market-hours logs found for this date in the current server window.")

with diag_tab:
    st.write({"log_coverage":{"requested_lines":log_limit,"parsed_rows":len(rows),"market_rows":len(today_rows),"complete":coverage_complete,"first_market_timestamp":str(first_ts) if first_ts else None,"error":log_error},"livez":{"http":live_code,"data":livez,"error":live_err},"readyz":{"http":ready_code,"data":readyz,"error":ready_err},"health_trading":{"http":trading_code,"data":trading,"error":trading_err}})

st.divider(); st.caption(f"Last checked: {datetime.now().astimezone().strftime('%d %b %Y, %I:%M:%S %p %Z')}")
