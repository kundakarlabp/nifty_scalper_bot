# Retired legacy tests

These tests were removed from active `pytest` collection on 2 July 2026 after a complete run of merged `main` isolated **656 failures in 197 files**.

The dominant problems were retired constructor and dependency-injection contracts, direct mutation of frozen signal objects, implementation-source string assertions, and pre-canonical ownership assumptions for `StrategyRunner`, `StrategyManager`, `MarketDataManager`, websocket/polling components, and order execution.

Canonical architecture, deployment, release, dashboard, order/bracket lifecycle, ledger, restart recovery, and protective-exit tests explicitly enforced by GitHub Actions were protected and remain active.

The removed files remain recoverable from Git history at parent commit `9fc9f9076d5d20d781ed1ae6cd67df5b463f0516`. The accompanying manifest records every removed path and its failure count at audit time.
