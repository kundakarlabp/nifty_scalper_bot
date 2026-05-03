import ast
from pathlib import Path


FORBIDDEN = {'historical_data', 'get_ohlc', 'fetch_history'}
TARGETS = {'on_tick', 'process_tick', '_process_queued_tick', '_emit_tick', 'on_datahub_tick', '_on_tick'}


def _assert_no_forbidden_calls(path: Path) -> None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in TARGETS:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if isinstance(fn, ast.Attribute):
                        name = fn.attr
                    elif isinstance(fn, ast.Name):
                        name = fn.id
                    else:
                        continue
                    assert name not in FORBIDDEN, f'{path}:{node.name} calls {name}'


def test_no_history_fetch_calls_in_tick_path() -> None:
    _assert_no_forbidden_calls(Path('src/nifty_scalper_bot/data/market_data_manager.py'))
    _assert_no_forbidden_calls(Path('src/nifty_scalper_bot/strategies/runner.py'))
