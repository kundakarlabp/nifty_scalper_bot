# Alert handler hygiene

This change targets the alert bridge that sends aggregated operator messages. It drops the generic PTB updater polling line at the alert-handler boundary while preserving classified internal Telegram events.
