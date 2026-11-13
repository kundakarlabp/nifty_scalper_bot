# Breeze historical research: one-command recovery and existing runner

**Current evidence:** The 25 original ZIPs in the private Drive folder are intact (107,821,151 bytes, 25/25 outer SHA-256 pass); they contain CMS-encrypted bytes. The original recipient key is not available in the current connected workspace, so the historical 1-minute contents have not been converted or replayed here. The separate validated EOD SQLite catalogue (494 F&O sessions) is available but contains no intraday quotes. No paid NSE data is necessary for the existing EOD archive.

## One command on the trusted owner machine

Keep the matching old RSA private key on the machine, not in ChatGPT, GitHub or Google Drive. The *existing* `scripts.finish_breeze_private_replay` locates it using the public certificate fingerprint and, after successful decryption, validates all inner file checksums, converts the windows into a private intraday SQLite, reconciles the original 1,454,867 1m observations, stages strictly eligible sessions and invokes the existing `strategy_research.run_archived_research` in shadow mode.

```sh
EXECUTION_MODE=SHADOW ENABLE_LIVE=false ALLOW_REAL_BROKER=false \
python -m scripts.finish_breeze_private_replay \
  --archives /private/owner/Breeze-original-25-ZIPs \
  --certificate deploy/breeze_transport_recipient.pem \
  --key-search-dir "$HOME/Documents" \
  --key-search-dir "$HOME/Downloads" \
  --output-root /private/nifty-research-run-20261113 \
  --eod-db /private/nifty_eod_reference.sqlite \
  --permit-partial --run-strategies
```

If the matching private key was lost but **verified plaintext** history was already extracted in a trusted private folder, use this same entrypoint without any CMS key:

```sh
EXECUTION_MODE=SHADOW ENABLE_LIVE=false ALLOW_REAL_BROKER=false \
python -m scripts.finish_breeze_private_replay \
  --extracted-windows /private/owner/extracted-Breeze-25-windows \
  --output-root /private/nifty-research-run-plaintext \
  --eod-db /private/nifty_eod_reference.sqlite \
  --permit-partial --run-strategies
```

The EOD catalogue must be extracted beforehand from the previously verified `NIFTY_EOD_Research_SQLite_20261112.sqlite.gz` file. The source ZIPs must be locally available only once; **do not duplicate the Google Drive download**. The research output root must not exist beforehand. Source archives and extracted data are never overwritten.

## How the existing strategy runner is reused

The bridge creates per-day `candles/*.json` files in the format already accepted by `nifty_scalper_bot.backtesting.strategy_research.load_archive`. It chooses a futures contract and matched CE/PE pair with **previous-session EOD contract evidence** and the previous available index close, never same-day EOD for the 09:15 basket. It keeps only synchronized, contiguous observed Breeze minutes. A partial session is allowed only with `--permit-partial`, and it requires a complete 15-minute opening range before invoking the existing ORB/VWAP/SMC strategy research. All gaps, excluded days and selected symbols are reported in `research_sessions/research_summary.json` and per-day `manifest.json` files.

This is a **bar-proxy research replay**, not an execution-grade NSE L2 fill simulator. The Breeze archives lack historical bid/ask depth, exact broker instrument-token history and observed broker fills. Generated results retain `execution_grade_ready=false` and `live_promotion_eligible=false`, regardless of simulated P&L. Production broker connections and risk controls are unchanged.

**The original key cannot be eliminated from old CMS ciphertext by deleting code or changing Drive sharing.** After a one-time recovery, future private history can be stored as checksum-verified unencrypted compressed datasets in a restricted Drive folder, without generating a new CMS key. This is not yet the output format of the existing GitHub acquisition workflows; do not assume those workflows have been changed.

## Why CMS encryption exists (verified 13 November 2026)

The source GitHub repository is **PUBLIC** (GitHub repository visibility API confirmed). GitHub Actions artifacts must not contain unencrypted private market data or credential-bearing raw files. The historical workflow used the public recipient certificate to encrypt those artifacts before upload for this reason. The user's private Google Drive permissions do not control public GitHub artifact access. Do **not** disable encryption on the public-repository GitHub workflows. Once decrypted on an owner-controlled machine, use access-restricted private Drive or private local disk for any unencrypted, SHA-256-verified working copy; never publish raw option data to the source repository. This is a configuration/security boundary, not an additional NSE data purchase.
