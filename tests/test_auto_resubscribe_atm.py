from __future__ import annotations

from src.data import source as src


def test_auto_resubscribe_atm_resubscribes_stale_tokens() -> None:
    class Dummy(src.DataSource):
        def __init__(self) -> None:
            self.atm_tokens = [1, 2]
            self.calls: list[list[int]] = []
            self._atm_next_check_ts = 0.0

        def ltp(self, token: int):  # type: ignore[override]
            return 100 if token == 2 else None

        def subscribe_tokens(self, tokens: list[int]) -> None:  # type: ignore[override]
            self.calls.append(tokens)

    ds = Dummy()
    ds.auto_resubscribe_atm()
    assert ds.calls == [[1]]
