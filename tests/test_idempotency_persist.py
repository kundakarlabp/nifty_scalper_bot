from src.logs.journal import Journal


def test_idempotency_persist(tmp_path):
    db = tmp_path / "journal.sqlite"
    j = Journal.open(str(db))
    j.append_event(
        ts="2024-01-01T00:00:00",
        trade_id="T1",
        leg_id="L1",
        etype="IDEMP",
        idempotency_key="KEY1",
        payload={},
    )
    j2 = Journal.open(str(db))
    assert j2.get_idemp_leg("KEY1") == "L1"
