from src.execution.order_executor import OrderExecutor
from src.logs.journal import Journal


def test_shutdown_appends_journal_line():
    journal = Journal.open(':memory:')
    exe = OrderExecutor(None, journal=journal)
    exe.shutdown()
    cur = journal._exec('SELECT trade_id, leg_id, etype FROM events')
    rows = cur.fetchall()
    assert ('SYSTEM', 'SHUTDOWN', 'SHUTDOWN') in rows
