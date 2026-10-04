"""Five ORB research runs must remain one-change experiments."""

from scripts.research_orb_iterative import BASE_OVERRIDES, candidates


def test_five_runs_change_exactly_one_orb_setting_from_lifecycle_reference():
    rows = candidates()
    assert [row["name"] for row in rows[:2]] == [
        "static_reference_075_25",
        "lifecycle_reference_075_25",
    ]
    experiments = rows[2:]
    assert len(experiments) == 5
    assert all(row["lifecycle_proxy"] is True for row in experiments)
    for row in experiments:
        extra = {
            key: value
            for key, value in row["overrides"].items()
            if BASE_OVERRIDES.get(key) != value
        }
        assert len(extra) == 1
