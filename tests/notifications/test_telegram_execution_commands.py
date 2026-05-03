from nifty_scalper_bot.notifications.telegram_commands import cmd_emergencystop

class S: pass

def test_emergency_stop_fallback():
    s=S(); s.order_manager=None
    assert 'unavailable' in cmd_emergencystop(None,None,s).lower()
