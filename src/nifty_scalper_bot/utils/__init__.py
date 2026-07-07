from __future__ import annotations


def _install_ptb_update_hook() -> None:
    try:
        from nifty_scalper_bot.notifications.updater_error_callback_patch import apply_patch

        apply_patch()
    except Exception:
        return


_install_ptb_update_hook()
