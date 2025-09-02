from src.strategies.scalping_strategy import EnhancedScalpingStrategy


def test_normalize_min_score_accepts_percentages() -> None:
    norm = EnhancedScalpingStrategy._normalize_min_score
    assert norm(0.35) == 0.35
    assert norm(35) == 0.35
    assert norm(70) == 0.70
