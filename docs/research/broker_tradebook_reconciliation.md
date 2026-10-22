# Broker tradebook reconciliation (read-only)

This process compares recorded closed trades against an explicitly exported Kite/Zerodha tradebook. It makes no broker API calls, places no orders, and does not change LIVE gates, risk budgets, or stops.

## Input

Use a CSV file on an authorized local machine with these headers:

- trade_id: **broker trade/fill ID**, not the bot lifecycle TRD identifier.
- order_id: exact broker order ID.
- tradingsymbol (or symbol): exchange trading symbol.
- transaction_type (or side): BUY / SELL.
- quantity: the filled quantity of this tradebook fill.
- average_price (or price): the fill price.

Each broker fill ID must be unique. Preserve split fills as separate rows. **Do not upload broker CSVs or credentials to GitHub.**

Use the existing canonical journal database and run:

    python scripts/reporting/analyze_completed_trades.py --trades-db /authorized/path/trade_journal.db --broker-trades-csv /authorized/path/kite_tradebook.csv > /authorized/path/trade_quality.json

The new broker_tradebook_reconciliation section reports counts and per-trade mismatch reasons. It requires exact entry/exit order IDs; it never matches using symbols and timestamps alone. Multi-leg exits with more than one exit order remain **unverified** unless their complete closing fill identity is available. Missing broker fills also remain **unverified**, not "no order placed".

The report verifies only that the exported fill prices and quantities are consistent with the local ledger, within a 0.05 INR-per-option-unit price tolerance. Export authenticity and completeness are external responsibilities. A virtual contract note is a broker *estimate*, not a settled daily contract note. This report therefore always leaves final_net_pnl_broker_reconciled=false; it must not be used to claim broker-reconciled profits, promote a strategy, disarm an open position, or accelerate a deployment.

Keep official broker contract notes and transaction-charge details separately for final financial reconciliation. Historical nonclosed rows are not evidence of live positions.
