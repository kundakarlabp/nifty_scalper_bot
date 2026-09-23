# 23 September 2026: broker P&L diagnostic

The broker order and position screenshots show three closed one-lot NIFTY option
trades: 23350 PE at 09:37 (99.90 to 96.10, -247 gross), 23350 PE at 09:53
(105.35 to 107.35, +130 gross), and 23400 CE at 10:18 (125.15 to 126.35,
+78 gross). Their sum is -39 gross, matching the broker positions screen and
the local strategy ledger. Actual contract-note charges are not in the screenshots.

The live log relay starts at 09:53, so the 09:37 strategy decision, ticks and
stop trajectory cannot be reconstructed from this window. The later two closed
brackets record MFE of 1.35R and 1.03R and exits of 0.25R and 0.04R. Both
stop-breach events used the LTP fallback when their cached depth was stale.
These observations justify a tick-level exit replay, not a live trailing
threshold change from three trades.

The existing P&L snapshot already collects gross NIFTY MIS day-position P&L,
but its diagnostic compared the local ledger only with equity margins
`m2m_realised`, which remained zero during these closed trades. When day-position
marked and closed totals agree with local closed gross P&L while margins disagree, the
diagnostic now reports `source_disagreement` and exposes the positions-versus-
strategy difference. Margins and positions remain separate observations; risk
continues to use the confirmed local ledger on any non-`matched` status.
