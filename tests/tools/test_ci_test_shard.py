from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "scripts" / "ci_test_shard.py"
    spec = importlib.util.spec_from_file_location("ci_test_shard_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shards_cover_each_discovered_test_file_exactly_once() -> None:
    module = _load_module()
    files = module.discover_test_files(ROOT)

    assert len(files) >= 500

    shards = [
        module.shard_files(
            files,
            shard_index=index,
            shard_count=module.DEFAULT_SHARD_COUNT,
        )
        for index in range(module.DEFAULT_SHARD_COUNT)
    ]
    combined = [path for shard in shards for path in shard]

    assert len(combined) == len(files)
    assert set(combined) == set(files)
    assert len(combined) == len(set(combined))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_discovery_uses_pytest_file_naming_only(tmp_path: Path) -> None:
    module = _load_module()
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_alpha.py").write_text("", encoding="utf-8")
    (tests / "beta_test.py").write_text("", encoding="utf-8")
    (tests / "helper.py").write_text("", encoding="utf-8")
    (tests / "notes.txt").write_text("", encoding="utf-8")

    assert module.discover_test_files(tmp_path) == (
        "tests/beta_test.py",
        "tests/test_alpha.py",
    )


def test_shard_partition_is_stable_and_balanced() -> None:
    module = _load_module()
    files = tuple(f"tests/test_{index:03d}.py" for index in range(17))

    shards = [
        module.shard_files(files, shard_index=index, shard_count=4)
        for index in range(4)
    ]

    assert shards[0] == files[0::4]
    assert shards[1] == files[1::4]
    assert shards[2] == files[2::4]
    assert shards[3] == files[3::4]
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_weighted_shards_balance_source_weight_deterministically() -> None:
    module = _load_module()
    files = tuple(f"tests/test_{index}.py" for index in range(8))
    weights = {
        files[0]: 100,
        files[1]: 90,
        files[2]: 80,
        files[3]: 70,
        files[4]: 60,
        files[5]: 50,
        files[6]: 40,
        files[7]: 30,
    }

    shards = [
        module.shard_files(
            files,
            shard_index=index,
            shard_count=2,
            weights=weights,
        )
        for index in range(2)
    ]
    totals = [sum(weights[path] for path in shard) for shard in shards]

    assert totals == [260, 260]
    assert [path for shard in shards for path in shard]
    assert set(shards[0]).isdisjoint(shards[1])


def test_invalid_shard_arguments_fail_closed() -> None:
    module = _load_module()
    files = ("tests/test_one.py",)

    for shard_index, shard_count in ((0, 0), (-1, 4), (4, 4)):
        try:
            module.shard_files(
                files,
                shard_index=shard_index,
                shard_count=shard_count,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid shard accepted: index={shard_index} count={shard_count}"
            )
