from pathlib import Path

from nifty_scalper_bot.data.bracket_store import BracketStore


def test_save_bracket_generates_virtual_order_id(tmp_path: Path) -> None:
    store = BracketStore(db_path=str(tmp_path / 'state.db'))
    payload = {'symbol': 'NFO:NIFTY24APR22500CE', 'entry_price': 100.0}

    store.save_bracket(payload)

    rows = store.load_all_brackets()
    assert rows
    assert rows[0]['order_id'].startswith('VIRTUAL-')
