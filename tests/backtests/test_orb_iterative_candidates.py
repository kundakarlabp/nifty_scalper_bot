"""Five ORB research runs must remain one-change experiments."""

from scripts.research_orb_iterative import (
    BASE_OVERRIDES,
    _selected_candidates,
    candidates,
    quality_score_8_candidate,
    score_v2_candidate,
)


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


def test_quality_score_8_candidate_is_focused_and_preserves_canonical_five():
    canonical_names = [row["name"] for row in candidates()]
    assert "quality_score_8" not in canonical_names
    row = quality_score_8_candidate()
    assert row["overrides"]["ORB_QUALITY_MIN_SCORE_SHADOW"] == "8.0"
    assert _selected_candidates(["quality_score_8"]) == [row]


def test_score_v2_candidate_is_frozen_research_only_profile():
    row = score_v2_candidate()
    assert row["name"] == "score_v2_dev_2017_2018"
    assert row["lifecycle_proxy"] is True
    assert row["research_quality_profile"] == "score_v2_dev_2017_2018"
    assert _selected_candidates(["score_v2_dev_2017_2018"]) == [row]
