# File: src/nifty_scalper_bot/brokers/instrument_lookup.py
# (drop into the brokers/ package). Lines numbered for patch reference.

1  from __future__ import annotations
2  import pandas as pd
3  import threading
4  import time
5  from dataclasses import dataclass
6  from typing import Dict, Tuple, Optional
7
8  @dataclass
9  class Instrument:
10     exchange: str
11     tradingsymbol: str
12     instrument_token: int
13     name: Optional[str] = None
14     expiry: Optional[str] = None  # YYYY-MM-DD
15     strike: Optional[float] = None
16     option_type: Optional[str] = None  # 'CE' or 'PE' or None
17
18 class InstrumentLookup:
19     """
20     Robust lookup for Kite instrument tokens.
21     Usage:
22         il = InstrumentLookup('path/to/instruments.csv')
23         token = il.get_token(exchange='NFO', tradingsymbol='NIFTY24DEC17400CE')
24         # or
25         token = il.get_token_by_fields(
26             exchange='NFO', underlying='NIFTY', expiry='2024-12-19',
27             strike=17400, option_type='CE'
28         )
29     """
30
31     def __init__(self, csv_path: str, ttl_seconds: int = 600):
32         self.csv_path = csv_path
33         self.ttl_seconds = int(ttl_seconds)
34         self._lock = threading.RLock()
35         self._loaded_at = 0.0
36         # Primary maps
37         self._by_exchange_symbol: Dict[Tuple[str, str], Instrument] = {}
38         # Secondary map for options: (exchange, underlying, expiry, strike, option_type)
39         self._by_option_fields: Dict[Tuple[str, str, str, float, str], Instrument] = {}
40         self.reload()
41
42     def _normalize_expiry(self, expiry) -> Optional[str]:
43         if expiry is None:
44             return None
45         # Accept either YYYY-MM-DD or other common formats; ensure YYYY-MM-DD
46         if isinstance(expiry, str) and len(expiry) == 10 and expiry[4] == '-':
47             return expiry
48         # try parse
49         try:
50             dt = pd.to_datetime(expiry, utc=True).date()
51             return dt.isoformat()
52         except Exception:
53             return None
54
55     def reload(self):
56         """Load or reload instruments CSV into memory."""
57         with self._lock:
58             df = pd.read_csv(self.csv_path, dtype=str, low_memory=False)
59             # ensure consistent column names (common variants)
60             df_cols = {c.lower(): c for c in df.columns}
61             # ensure required columns exist
62             expected = ['exchange', 'tradingsymbol', 'instrument_token']
63             for col in expected:
64                 if col not in (c.lower() for c in df.columns):
65                     raise RuntimeError(f"instruments CSV missing required column: {col}")
66
67             self._by_exchange_symbol.clear()
68             self._by_option_fields.clear()
69
70             # Normalize and populate
71             for _, row in df.iterrows():
72                 exch = str(row['exchange']).strip()
73                 sym = str(row['tradingsymbol']).strip()
74                 token = int(row['instrument_token'])
75                 name = row.get('name') if 'name' in row else None
76                 expiry = None
77                 strike = None
78                 opt_type = None
79
80                 # try to parse expiry/strike/option_type from available columns
81                 # Kite instrument CSV often contains: expiry, strike, instrument_type (CE/PE)
82                 # We'll defensively check multiple column names.
83                 rlower = {k.lower(): v for k, v in row.items()}
84                 # expiry column
85                 for col in ('expiry', 'expiry_date', 'expiry_dt'):
86                     if col in rlower and pd.notna(rlower[col]):
87                         expiry = self._normalize_expiry(rlower[col])
88                         break
89                 # strike
90                 for col in ('strike', 'strike_price'):
91                     if col in rlower and pd.notna(rlower[col]):
92                         try:
93                             strike = float(rlower[col])
94                         except Exception:
95                             strike = None
96                         break
97                 # option type
98                 for col in ('instrument_type', 'option_type', 'segment'):
99                     if col in rlower and pd.notna(rlower[col]):
100                         # instrument_type often 'CE'/'PE' for options
101                         val = str(rlower[col]).upper()
102                         if val in ('CE', 'PE'):
103                             opt_type = val
104                             break
105
106                 inst = Instrument(exchange=exch, tradingsymbol=sym,
107                                   instrument_token=token, name=name,
108                                   expiry=expiry, strike=strike, option_type=opt_type)
109
110                 # exact exchange+tradingsymbol map (primary)
111                 key = (exch.upper(), sym.upper())
112                 self._by_exchange_symbol[key] = inst
113
114                 # populate option-fields map if we have necessary pieces
115                 if opt_type and expiry and strike is not None:
116                     # underlying: attempt to infer from tradingsymbol prefix (NIFTY, BANKNIFTY, etc)
117                     # convention: tradingsymbol often starts with underlying, e.g., NIFTY24DEC17400CE
118                     under = sym.upper()
119                     # Try to extract base underlying (non-numeric prefix)
120                     # crude heuristic: take all leading letters until a digit is found
121                     base = ''
122                     for ch in under:
123                         if ch.isalpha():
124                             base += ch
125                         else:
126                             break
127                     base = base or under
128                     opt_key = (exch.upper(), base, expiry, float(strike), opt_type.upper())
129                     self._by_option_fields[opt_key] = inst
130
131             self._loaded_at = time.time()
132
133     def _ensure_fresh(self):
134         if time.time() - self._loaded_at > self.ttl_seconds:
135             self.reload()
136
137     def get_token(self, exchange: str, tradingsymbol: str) -> int:
138         """Get token by exact exchange+tradingsymbol. Raises KeyError if not found."""
139         with self._lock:
140             self._ensure_fresh()
141             key = (exchange.upper(), tradingsymbol.upper())
142             inst = self._by_exchange_symbol.get(key)
143             if inst is None:
144                 raise KeyError(f"No instrument for exchange={exchange} tradingsymbol={tradingsymbol}")
145             return inst.instrument_token
146
147     def get_token_by_fields(self, exchange: str, underlying: str, expiry: str,
148                             strike: float, option_type: str) -> int:
149         """Lookup by canonical option fields. expiry must be YYYY-MM-DD or parseable."""
150         with self._lock:
151             self._ensure_fresh()
152             expiry_norm = self._normalize_expiry(expiry)
153             key = (exchange.upper(), underlying.upper(), expiry_norm, float(strike), option_type.upper())
154             inst = self._by_option_fields.get(key)
155             if inst:
156                 return inst.instrument_token
157             # deterministic fallback: attempt to find by scanning entries for the expiry/strike/type
158             for k, v in self._by_option_fields.items():
159                 (exch, base, exp, st, ot) = k
160                 if exch == exchange.upper() and exp == expiry_norm and st == float(strike) and ot == option_type.upper():
161                     # return the first exact numerical match (should be deterministic)
162                     return v.instrument_token
163             raise KeyError(f"No option instrument for {exchange} {underlying} {expiry_norm} {strike} {option_type}")
