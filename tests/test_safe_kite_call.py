import logging

def test_safe_kite_call_permission_logging(monkeypatch, caplog):
    from src.execution import order_executor as oe

    oe._PERM_WARN_TS = 0.0
    ticks = [301.0]

    def fake_time():
        return ticks[0]

    monkeypatch.setattr(oe.time, "time", fake_time)

    def boom():
        raise oe.PermissionException("no perms")

    logger = logging.getLogger("perm")
    with caplog.at_level(logging.WARNING, logger="perm"):
        assert oe._safe_kite_call(boom, 1, "perm", logger) == 1
        first = len(caplog.records)
        assert first == 1
        ticks[0] = 310.0
        assert oe._safe_kite_call(boom, 1, "perm", logger) == 1
        assert len(caplog.records) == first
        ticks[0] = 650.0
        assert oe._safe_kite_call(boom, 1, "perm", logger) == 1
        assert len(caplog.records) == first + 1
