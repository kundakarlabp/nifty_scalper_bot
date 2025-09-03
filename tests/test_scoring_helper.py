from src.strategies.scoring import compute_score


def test_compute_score_basic() -> None:
    weights = {"a": 0.5, "b": 1.0}
    features = {"a": 2.0, "b": -1.5, "c": 10.0}
    si = compute_score(weights, features)
    assert si.total == -0.5
    assert si.items == {"a": 1.0, "b": -1.5}

