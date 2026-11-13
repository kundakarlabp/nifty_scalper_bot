# Offline NIFTY historical backtesting bridge

The existing 25 Breeze archives use **CMS envelope encryption**. They cannot be unencrypted without the matching *original* private key. Google Drive account authorization allows downloading the encrypted files but does not replace that cryptographic key. This software never requests or uploads the key; it runs on the owner's trusted machine.

Once the original key is located (or a separately extracted and verified plaintext copy is available), the existing merged tools are already sufficient:

```bash
# All commands in a separate, private, research-only checkout. The key stays on this machine.
python -m scripts.unseal_breeze_private_windows \
  --encrypted-windows /private/Breeze-archives \
  --private-key /private/owner-original-cms-key.pem \
  --recipient-cert deploy/breeze_transport_recipient.pem \
  --output /private/breeze/extracted --expected-windows 25

python -m scripts.prepare_breeze_research_dataset build \
  --extracted-windows /private/breeze/extracted \
  --database /private/breeze/nifty_1m.sqlite \
  --summary /private/breeze/coverage_summary.json --expected-windows 25

# New adapter stages historically identified index, futures and CE+PE pairs, checking
# the prior-session EOD master and synchronized intraday market-clock coverage.
EXECUTION_MODE=SHADOW ENABLE_LIVE=false ALLOW_REAL_BROKER=false \
python -m scripts.stage_breeze_sqlite_research \
  --breeze-db /private/breeze/nifty_1m.sqlite \
  --summary /private/breeze/coverage_summary.json \
  --eod-db /private/breeze/nifty_eod_reference.sqlite \
  --output-root /private/breeze/offline-run \
  --permit-partial --run-strategies
```

`nifty_eod_reference.sqlite` is the previously verified, unencrypted EOD research catalogue. Each selected futures/option instrument must have appeared in the **previous session's** EOD contract master, with matching historical expiry/lot and same-day observed Breeze candles. The pairing uses prior-day index close to avoid same-day EOD/close lookahead. This is an **ex-post research cohort**, not proof that a broker could execute it. No instrument tokens or option quotes are manufactured; the canonical strategy research function is reused, but live orders and production risk gates are never called.

Without `--permit-partial`, only sessions containing every synchronized minute for the identified four instruments are staged. With `--permit-partial`, the longest continuous segment of >=60 complete aligned minutes is preserved with explicit omissions and only replays when the 15-minute session opening is intact. `research_summary.json` reports all rejected days. Summary SHA-256 and SQLite row counts must match. A no-data or no-key state is **not** a completed historical backtest.

For new captures, use private, owner-only compressed research archives (plus SHA-256 and access-control checks) *instead of new CMS encryption*, but do not delete old ciphertext or change its provenance; original missing private keys cannot be replaced retrospectively. The GitHub research writer still encrypts by default unless its future capture transport is separately changed and validated. No plaintext market data or private key should be committed to the repository.