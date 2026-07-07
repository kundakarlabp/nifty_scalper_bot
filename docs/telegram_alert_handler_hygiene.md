# Telegram alert handler hygiene

The operator alert bridge is the direct source of aggregated alert messages. The Telegram package logger filter alone is not enough if the root alert handler receives a propagated third-party record. The alert bridge now drops only the generic PTB updater polling line and keeps classified internal Telegram events visible.
