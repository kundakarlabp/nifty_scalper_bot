"""Read-only command and observability console for Nifty Scalper Bot."""
from __future__ import annotations
import csv, io, os, re
from collections import Counter
from datetime import date, datetime, time
from typing import Any
import requests
import streamlit as st

st.set_page_config(page_title="Nifty Scalper Command Center",page_icon="📈",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""<style>.block-container{padding-top:.8rem;padding-bottom:2rem;max-width:1350px}div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);border-radius:12px;padding:.55rem .7rem}div[data-testid="stMetricValue"]{font-size:1.18rem}.small{opacity:.75;font-size:.85rem}@media(max-width:640px){.block-container{padding-left:.5rem;padding-right:.5rem}h1{font-size:1.45rem!important}}</style>""",unsafe_allow_html=True)
LOG_RE=re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) IST\s+(?P<message>.*)$")
FUNNEL={"Evaluations":("RUNNER","evaluation"),"Signals":("SIGNAL_GENERATED","signal generated"),"Candidates":("CANDIDATE","candidate generated"),"Approved":("APPROVED_CANDIDATE","approved candidate"),"Orders":("ORDER_SENT","Sending Order","TRADE_ATTEMPT"),"Fills":("FILLED","ORDER_COMPLETE"),"Closed":("TRADE_CLOSED","EXIT")}
BLOCK_TERMS=("BLOCK","REJECT","SKIP","DENIED","SUPPRESS","NOT_READY","UNAVAILABLE")
IMPORTANT=("ERROR","EXCEPTION","RESTART","WEBSOCKET","DISCONNECT","RECONCIL","ORDER_","TRADE_","SIGNAL_","CANDIDATE","BLOCK")

def secret(n,d=""):
    try:v=st.secrets.get(n,d)
    except Exception:v=d
    return str(v or os.getenv(n,d)).strip()
def base():return secret("BOT_API_URL").rstrip("/")
def headers():
    t=secret("BOT_DASHBOARD_TOKEN");return {"Authorization":f"Bearer {t}"} if t else {}
def get_json(path):
    if not base():return None,"BOT_API_URL not configured",None
    try:
        r=requests.get(f"{base()}{path}",headers=headers(),timeout=8);c=r.status_code;r.raise_for_status();x=r.json();return (x if isinstance(x,dict) else None),(None if isinstance(x,dict) else "Unexpected response"),c
    except Exception as e:return None,str(e),getattr(getattr(e,"response",None),"status_code",None)
def fetch_logs(lines):
    if not base():return [],"BOT_API_URL not configured",False
    try:r=requests.get(f"{base()}/admin/logs/download",params={"fmt":"txt","lines":lines},headers=headers(),timeout=30);r.raise_for_status()
    except Exception as e:return [],f"Log download failed: {e}",False
    raw=r.text.splitlines();out=[]
    for line in raw:
        m=LOG_RE.match(line.strip())
        if not m:continue
        try:ts=datetime.strptime(m.group("timestamp"),"%Y-%m-%d %H:%M:%S")
        except ValueError:continue
        out.append((ts,m.group("message")))
    return out,None,len(raw)<lines
def day_rows(rows,d):
    lo=datetime.combine(d,time(9,15));hi=datetime.combine(d,time(15,30));return [(t,m) for t,m in rows if lo<=t<=hi]
def funnel(rows):return {k:sum(1 for _,m in rows if any(x.lower() in m.lower() for x in terms)) for k,terms in FUNNEL.items()}
def blocks(rows):
    c=Counter()
    for _,msg in rows:
        if not any(x in msg.upper() for x in BLOCK_TERMS):continue
        reason="other"
        for p in (r"(?:reason|blocker|gate)[=: ]+([A-Za-z0-9_.-]+)",r"([A-Z][A-Z0-9_]{3,}(?:BLOCK|REJECT|SKIP)[A-Z0-9_]*)"):
            z=re.search(p,msg,re.I)
            if z:reason=z.group(1).lower();break
        c[reason]+=1
    return c
def label(v):
    if v is None or v=="":return "—"
    if isinstance(v,bool):return "YES" if v else "NO"
    return str(v)
def csv_data(rows):
    b=io.StringIO();w=csv.writer(b);w.writerow(["timestamp_ist","message"]);w.writerows([(f"{t:%Y-%m-%d %H:%M:%S} IST",m) for t,m in rows]);return b.getvalue().encode("utf-8-sig")
def conversion(a,b):return f"{(100*b/a):.1f}%" if a else "—"

st.title("Nifty Scalper — Command & Observability Center")
st.caption("Canonical read-only operational console • broker/execution path remains isolated")
with st.sidebar:
    st.subheader("Console");st.code(base() or "BOT_API_URL not set",language=None);limit=st.selectbox("Log window",[20000,50000],index=1)
    if st.button("Refresh",use_container_width=True):st.rerun()

live,le,lc=get_json("/livez");ready,re,rc=get_json("/readyz");trade,te,tc=get_json("/health/trading")
broker=(trade or {}).get("broker") or {};recon=(trade or {}).get("reconciliation") or {};state=(trade or {}).get("state") or {};selected=(trade or {}).get("selected") or {};hist=(trade or {}).get("history") or {};pressure=(trade or {}).get("tick_pressure") or {}
blockers=(trade or {}).get("blockers") or (ready or {}).get("blockers") or []
rows,log_error,below_cap=fetch_logs(limit);chosen=st.date_input("Session",value=date.today(),max_value=date.today());market=day_rows(rows,chosen);first=market[0][0] if market else None;complete=bool(market and below_cap and first.time()<=time(9,16));counts=funnel(market);bc=blocks(market)

# Command strip: fastest answer to whether the system can trade.
strip=st.columns(6)
strip[0].metric("ENGINE","HEALTHY" if live else "DOWN")
strip[1].metric("BROKER","READY" if broker.get("ready") else "CHECK")
data_ready=state.get("data_hard_ready",(trade or {}).get("data_hard_ready"));strip[2].metric("DATA","READY" if data_ready else "CHECK")
strip[3].metric("EXECUTION","ARMED" if (trade or {}).get("live_orders_armed") else "NOT ARMED")
strip[4].metric("POSITION","SAFE" if not recon.get("unprotected_positions") else "UNPROTECTED")
strip[5].metric("BUILD",str((trade or {}).get("build_sha") or (live or {}).get("build_sha") or "—")[:8])
if not live:st.error("Engine API unreachable")
elif recon.get("unprotected_positions"):st.error("Unprotected broker position detected — inspect immediately")
elif blockers:st.warning("Primary current blocker: "+str((trade or {}).get("primary_blocker") or blockers[0]))
elif (trade or {}).get("live_orders_armed"):st.success("Execution path reports LIVE and armed")
else:st.info("Engine online; live orders are not currently armed")

command_tab,why_tab,funnel_tab,trades_tab,data_tab,timeline_tab,review_tab,logs_tab=st.tabs(["Command Center","Why no trade?","Signal Funnel","Trades","Market & Data","Timeline","Day Review","Raw Logs"])
with command_tab:
    c=st.columns(4);c[0].metric("Balance",label(broker.get("balance")));c[1].metric("Execution ready",label((trade or {}).get("ready")));c[2].metric("Reconciled",label(recon.get("completed")));c[3].metric("Log coverage","COMPLETE" if complete else "PARTIAL")
    st.subheader("Current blockers")
    if blockers:
        for b in blockers:st.warning(str(b))
    else:st.success("No blocker reported by readiness endpoints")
    st.subheader("Selected contracts")
    x=st.columns(3);x[0].metric("ATM",label(selected.get("atm")));x[1].metric("CE",label(selected.get("ce")));x[2].metric("PE",label(selected.get("pe")))
with why_tab:
    st.subheader("Why isn't it trading?")
    if bc:
        total=sum(bc.values());table=[{"rank":i+1,"reason":k,"events":v,"share":f"{100*v/total:.1f}%"} for i,(k,v) in enumerate(bc.most_common(25))];st.dataframe(table,use_container_width=True,hide_index=True)
        st.caption("Ranked from log-derived suppression/rejection events. Current readiness blockers above remain authoritative for present state.")
    elif blockers:
        st.warning("Current endpoint blocker: "+"; ".join(map(str,blockers)))
    else:st.info("No suppression reason found in the available session window.")
with funnel_tab:
    cols=st.columns(len(counts))
    for col,(k,v) in zip(cols,counts.items()):col.metric(k,f"{v:,}")
    names=list(counts);st.subheader("Stage conversion")
    conv=[]
    for a,b in zip(names,names[1:]):conv.append({"from":a,"to":b,"conversion":conversion(counts[a],counts[b])})
    st.dataframe(conv,use_container_width=True,hide_index=True)
with trades_tab:
    trade_events=[(t,m) for t,m in market if any(x in m.upper() for x in ("ORDER_","TRADE_","FILLED","EXIT"))]
    st.metric("Order / trade events",len(trade_events))
    for t,m in trade_events[-300:][::-1]:st.text(f"{t:%H:%M:%S}  {m}")
    if not trade_events:st.info("No order/trade lifecycle events in available logs.")
with data_tab:
    c=st.columns(4);c[0].metric("Data hard ready",label(data_ready));c[1].metric("Pending ticks",label(pressure.get("pending_ticks")));c[2].metric("Dropped ticks",label(pressure.get("dropped_total")));c[3].metric("Accounting balanced",label(pressure.get("accounting_balanced")))
    st.subheader("History readiness");st.json(hist if hist else {"status":"not exposed by this endpoint"})
with timeline_tab:
    sig=[(t,m) for t,m in market if any(x in m.upper() for x in IMPORTANT)];st.caption(f"{len(sig):,} significant events")
    for t,m in sig[-500:][::-1]:st.text(f"{t:%H:%M:%S}  {m}")
with review_tab:
    st.subheader(chosen.isoformat());c=st.columns(4);c[0].metric("Timestamped rows",f"{len(market):,}");c[1].metric("Approved",counts.get("Approved",0));c[2].metric("Orders",counts.get("Orders",0));c[3].metric("Closed",counts.get("Closed",0))
    if complete:st.success("Available server window appears to cover market open through close.")
    else:st.warning("Historical review is partial until persistent Supabase daily archive ingestion is active.")
    st.caption("Day-to-day/build comparison will use the persistent Supabase archive once ingestion is populated; the trading engine never depends on it.")
with logs_tab:
    if log_error:st.error(log_error)
    q=st.text_input("Search logs",placeholder="RUNNER / ORDER / reason / symbol / error")
    shown=[r for r in market if not q or q.lower() in r[1].lower()]
    st.caption(f"Showing {len(shown):,} matching rows; display capped to latest 5,000.")
    st.text_area("Raw diagnostic log",value="\n".join(f"{t:%H:%M:%S} {m}" for t,m in shown[-5000:]),height=520)
    st.download_button("Download available session CSV",csv_data(market),file_name=f"niftybot-{chosen.isoformat()}.csv",mime="text/csv",use_container_width=True)
    with st.expander("Endpoint diagnostics"):st.write({"coverage":{"complete":complete,"first":str(first) if first else None,"rows":len(market),"error":log_error},"livez":{"http":lc,"data":live,"error":le},"readyz":{"http":rc,"data":ready,"error":re},"trading":{"http":tc,"data":trade,"error":te}})
st.divider();st.caption("Read-only • "+datetime.now().astimezone().strftime("%d %b %Y %I:%M:%S %p %Z"))
