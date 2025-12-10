# 🔴 RAILWAY ENVIRONMENT VARIABLES - MISSING & REQUIRED

**Analysis Date**: 2025-12-10 20:54 IST  
**Status**: 🔴 BLOCKING - 15+ critical variables missing

---

## ✅ VARIABLES YOU HAVE SET (22)

```
1.  DATA__HIST_PATH
2.  ELITE_MAX_CONCURRENT_STRATEGIES
3.  ELITE_STRATEGIES_ENABLED
4.  ENABLE_LIVE
5.  KITE_ACCESS_TOKEN
6.  KITE_API_KEY
7.  KITE_API_SECRET
8.  LOG_LEVEL
9.  MARKET_DATA_MODE
10. OI_MAX_PAIN_ENABLED
11. ORDER_DEFAULT_TYPE
12. ORDER_LIMIT_OFFSET_PCT
13. PNL_PERSIST_PATH
14. POLL_INTERVAL_MS
15. POLL_REQUIRE_DEPTH
16. PYTHONPATH
17. REGIME_FILTER_BYPASS
18. SMC_ENABLED
19. SQLITE_DB_PATH
20. STALE_THRESHOLD_MS
21. TELEGRAM__BOT_TOKEN
22. TELEGRAM__CHAT_ID
23. TELEGRAM_PUBLIC_BASE_URL
24. TELEGRAM_WEBHOOK_ENABLED
25. VWAP_PRO_ENABLED
```

---

## 🔴 CRITICAL MISSING VARIABLES (Blocking Strategies)

### **TIER 1: Elite Strategy Confidence Thresholds** ← WHY STRATEGIES NOT LOADING!

These are **MISSING** and causing `elite_strategies` list to be empty:

```env
# FROM settings.py line 1235-1270 in _build_elite_settings()
# IF MISSING, all strategies use VERY HIGH defaults (70-75%+)
# This makes them inactive because thresholds are too strict

SMC_MIN_CONFIDENCE=40.0           # ← MISSING (default: 75.0)
VWAP_MIN_CONFIDENCE=40.0          # ← MISSING (default: 72.0)
OI_MIN_CONFIDENCE=40.0            # ← MISSING (default: 68.0)
GAMMA_MIN_CONFIDENCE=65.0         # ← MISSING (default: 65.0) - borderline
CPR_MIN_CONFIDENCE=68.0           # ← MISSING (default: 68.0) - borderline
ORDER_FLOW_MIN_CONFIDENCE=70.0    # ← MISSING (default: 70.0) - borderline
BB_MIN_CONFIDENCE=55.0            # ← MISSING (default: 55.0) - borderline
RSI_MIN_CONFIDENCE=40.0           # ← MISSING (default: 62.0)
ORB_MIN_CONFIDENCE=63.0           # ← MISSING (default: 63.0) - borderline
STRADDLE_MIN_CONFIDENCE=73.0      # ← MISSING (default: 73.0) - borderline
```

**IMPACT**: ❌ **ALL STRATEGY THRESHOLDS TOO HIGH** = No signals generated

---

### **TIER 2: Individual Strategy Enablement**

```env
# These should be SET to enable/disable individual strategies
# Missing = uses defaults from settings.py

SMC_ENABLED=true                  # ✅ You have: true (Line 1237)
VWAP_PRO_ENABLED=true             # ✅ You have: true (Line 1245)
OI_MAX_PAIN_ENABLED=true          # ✅ You have: true (Line 1253)

# These are COMPLETELY MISSING (causes silent disable):
GAMMA_ENABLED=false               # ← MISSING (default: false) - OK
CPR_ENABLED=true                  # ← MISSING (default: true) - Should add
ORDER_FLOW_ENABLED=true           # ← MISSING (default: true) - Should add
BB_SQUEEZE_ENABLED=true           # ← MISSING (default: true) - Should add
RSI_DIV_ENABLED=true              # ← MISSING (default: true) - Should add
ORB_ENABLED=true                  # ← MISSING (default: true) - Should add
STRADDLE_ENABLED=true             # ← MISSING (default: true) - Should add
```

**IMPACT**: 🟡 **Strategies still try to load but with high thresholds = no signals**

---

## 🟠 MISSING IMPORTANT VARIABLES (Affect Trading Quality)

```env
# RISK & POSITION MANAGEMENT (Critical for trading)
RISK_DAILY_LOSS_PCT=2.0           # ← MISSING (trades blocked if limit hit)
RISK_PER_TRADE_PCT=5.0            # ← MISSING (position sizing)
RISK_MAX_CONSEC_LOSSES=3          # ← MISSING (stops after 3 losses)
RISK_LOSS_COOLDOWN_SEC=90.0       # ← MISSING (cooldown after loss)
RISK_BREAKER_AUTO_SHADOW=true     # ← MISSING (auto switch to paper on loss)

# ORDER EXECUTION (Critical for execution speed)
ORDER_RETRY_ATTEMPTS=3            # ← MISSING (affects retries)
ORDER_RETRY_DELAY_SECONDS=0.5     # ← MISSING
ORDER_REPLACE_ATTEMPTS=3          # ← MISSING
ORDER_REPLACE_BACKOFF_SECONDS=0.75 # ← MISSING
ORDER_RATE_PER_SECOND=5           # ← MISSING (affects order rate limiting)
ORDER_RATE_PER_MINUTE=150         # ← MISSING
ORDER_RATE_PER_DAY=2000           # ← MISSING

# STREAMER & DATA (Critical for data flow)
STREAMER_GAP_SECONDS=5.0          # ← MISSING
STREAMER_BACKFILL_LIMIT=10        # ← MISSING
STREAMER_BACKFILL_INTERVAL=minute # ← MISSING
STREAMER_QUEUE_SIZE=1000          # ← MISSING

# SESSION MANAGEMENT
SESSION_ALLOW_OUT_OF_HOURS=false  # ← MISSING (but has default: false)
ALLOW_OFFHOURS_TESTING=false      # ← MISSING (alias)

# SHADOW TRADING (Paper trading config)
SHADOW_DRIFT_THRESHOLD_PCT=2.5    # ← MISSING
SHADOW_AUTO_DISABLE_LIVE=true     # ← MISSING
SHADOW_DRIFT_SUSTAIN_SECONDS=30.0 # ← MISSING

# REGIME GATES (Market analysis)
REGIME__ADX_MIN_TREND=25.0        # ← MISSING (affects trading signals)
REGIME__ATR_MULT_RANGE=1.0        # ← MISSING
NOTRADE__OPEN_MIN=15              # ← MISSING (skip first 15 min)
NOTRADE__PRECLOSE_MIN=30          # ← MISSING (skip last 30 min)

# SELECTOR (Strike selection)
SELECTOR__MODE=ATM                # ← MISSING (At-The-Money)
SELECTOR__EXPIRY=weekly           # ← MISSING
SELECTOR__DELTA_TARGET=0.3        # ← MISSING
SELECTOR__EXPIRY_ROLL_MINUTES=45  # ← MISSING
```

---

## 🟡 OPTIONAL BUT RECOMMENDED

```env
# FEATURE FLAGS
FEATURE_ORDER_WITHOUT_TOKEN=true  # ← MISSING (optional)
FEATURE_RESOLVER_LEARN_FROM_QUOTES=true # ← MISSING (optional)

# EXECUTION SETTINGS
EXECUTION__ENABLE_BRACKET_MANAGER=true    # ← MISSING (default: true)
EXECUTION__BRACKET_AUTO_REDUCE_SL=true    # ← MISSING (default: true)
EXECUTION__BRACKET_STALE_CLEANUP_SECONDS=86400 # ← MISSING (default: 86400)

# POLLING
POLL_BATCH_SIZE=50                # ← MISSING (default: 50)
POLL_INTERVAL_MS_JITTER_PCT=0.15  # ← MISSING (default: 0.15)

# TELEGRAM ADVANCED
TELEGRAM_RATE_PER_SECOND=25.0     # ← MISSING (msg rate limit)
TELEGRAM_BURST_CAPACITY=25.0      # ← MISSING
TELEGRAM_ALLOW_POLL_FALLBACK=false # ← MISSING (use polling if webhook fails)
TELEGRAM_ENABLED=true             # ← MISSING (check if token set)

# MONITORING
TELEMETRY_TAGS=production,nifty   # ← MISSING (optional)
```

---

## 📊 THE REAL PROBLEM

### **Why Elite Strategies Show "Loaded" but Do Nothing**

Look at the code flow in `settings.py` line 1235-1280:

```python
def _build_elite_settings() -> EliteStrategiesSettings:
    smc_cfg = SMCStrategyConfig(
        enabled=_env_bool("SMC_ENABLED", default=True),  # ✅ You set to true
        min_confidence=_env_float("SMC_MIN_CONFIDENCE", default=75.0, minimum=0.0),
        #                                               ↑ Uses 75.0 IF NOT SET
        # ...
    )
    # Same for VWAP, OI, GAMMA, etc.
```

**Your Railway config:**
- ✅ SMC_ENABLED = true → Strategy loads
- ❌ SMC_MIN_CONFIDENCE = MISSING → Uses default 75.0
- ❌ VWAP_MIN_CONFIDENCE = MISSING → Uses default 72.0
- ❌ OI_MIN_CONFIDENCE = MISSING → Uses default 68.0

**Result:**
- Strategies load and appear in the "loaded" log ✅
- But confidence thresholds are too strict (70-75%+)
- Market signals rarely reach 75% confidence
- **→ No actual trades executed** ❌

---

## ✅ IMMEDIATE FIX (Copy-Paste to Railway)

### **CRITICAL - Do This First**

```env
# CONFIDENCE THRESHOLDS (WHY STRATEGIES NOT WORKING)
SMC_MIN_CONFIDENCE=40.0
VWAP_MIN_CONFIDENCE=40.0
OI_MIN_CONFIDENCE=40.0
GAMMA_MIN_CONFIDENCE=65.0
CPR_MIN_CONFIDENCE=68.0
ORDER_FLOW_MIN_CONFIDENCE=70.0
BB_MIN_CONFIDENCE=55.0
RSI_MIN_CONFIDENCE=40.0
ORB_MIN_CONFIDENCE=63.0
STRADDLE_MIN_CONFIDENCE=73.0

# STRATEGY ENABLING (Make sure all are enabled)
GAMMA_ENABLED=false
CPR_ENABLED=true
ORDER_FLOW_ENABLED=true
BB_SQUEEZE_ENABLED=true
RSI_DIV_ENABLED=true
ORB_ENABLED=true
STRADDLE_ENABLED=true

# RISK & POSITION SIZING (Allows trading to happen)
RISK_DAILY_LOSS_PCT=2.0
RISK_PER_TRADE_PCT=5.0
RISK_MAX_CONSEC_LOSSES=3
RISK_LOSS_COOLDOWN_SEC=90.0
RISK_BREAKER_AUTO_SHADOW=true
MIN_LOTS_PER_TRADE=1
MAX_LOTS_PER_TRADE=3

# ORDER EXECUTION SETTINGS
ORDER_RETRY_ATTEMPTS=3
ORDER_RETRY_DELAY_SECONDS=0.5
ORDER_REPLACE_ATTEMPTS=3
ORDER_REPLACE_BACKOFF_SECONDS=0.75
ORDER_RATE_PER_SECOND=5
ORDER_RATE_PER_MINUTE=150
ORDER_RATE_PER_DAY=2000

# STREAMER & DATA
STREAMER_GAP_SECONDS=5.0
STREAMER_BACKFILL_LIMIT=10
STREAMER_BACKFILL_INTERVAL=minute
STREAMER_QUEUE_SIZE=1000

# SESSION MANAGEMENT
SESSION_ALLOW_OUT_OF_HOURS=false

# REGIME GATES (Market analysis)
REGIME__ADX_MIN_TREND=25.0
REGIME__ATR_MULT_RANGE=1.0
NOTRADE__OPEN_MIN=15
NOTRADE__PRECLOSE_MIN=30

# STRIKE SELECTOR
SELECTOR__MODE=ATM
SELECTOR__EXPIRY=weekly
SELECTOR__DELTA_TARGET=0.3
SELECTOR__EXPIRY_ROLL_MINUTES=45
SELECTOR__MONTHLY_HALT_MINUTES=30

# EXECUTION
EXECUTION__ENABLE_BRACKET_MANAGER=true
EXECUTION__BRACKET_AUTO_REDUCE_SL=true
EXECUTION__BRACKET_STALE_CLEANUP_SECONDS=86400

# TELEGRAM
TELEGRAM_RATE_PER_SECOND=25.0
TELEGRAM_BURST_CAPACITY=25.0
TELEGRAM_ALLOW_POLL_FALLBACK=false
```

---

## 🚀 DEPLOYMENT STEPS

### **Step 1: Add to Railway**
1. Go to Railway dashboard
2. Find your Nifty Scalper Bot project
3. Click Variables tab
4. **Paste the above 50+ lines exactly**
5. Save

### **Step 2: Redeploy** (Not just restart!)
```bash
# In Railway
1. Go to Deployments
2. Click "Redeploy" on latest deployment
3. Wait 3-5 minutes
```

### **Step 3: Verify**

Watch logs for:
```
✅ elite_strategies_loaded (count: 10)
✅ Condition met: settings_validated
```

Now you should see actual strategy signals:
```
⚡ VWAP CROSSOVER DETECTED
🟢 STRIKE SELECTED
🟢 ORDER SUBMITTED
```

---

## 📋 VERIFICATION CHECKLIST

- [ ] Added all 10 confidence threshold variables
- [ ] Added all 7 strategy enablement variables
- [ ] Added all 5 risk management variables
- [ ] Added all 7 order execution variables
- [ ] Added all 4 streamer variables
- [ ] Added regex/selector/execution variables
- [ ] Saved variables in Railway
- [ ] Clicked **Redeploy** (not restart)
- [ ] Waited 3-5 minutes for deployment
- [ ] Checked logs for "elite_strategies_loaded (count: 10)"
- [ ] Verified strategy signals appearing

---

## 🎯 EXPECTED OUTCOME

**Before Fix:**
```
✅ Condition met: elite_strategies_loaded
← No count shown, strategies inactive
```

**After Fix:**
```
✅ Condition met: elite_strategies_loaded (count: 10)
⚡ VWAP CROSSOVER DETECTED: NFO:NIFTY25DEC25750CE
🟢 ORDER SUBMITTED! ID: 123456789
```

---

**Last Updated**: 2025-12-10 20:54 IST

