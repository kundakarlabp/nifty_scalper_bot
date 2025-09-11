import pytest
from src.config import RiskSettings


def test_exposure_basis_validation_accepts_case_variants():
    assert RiskSettings(exposure_basis="Premium").exposure_basis == "premium"
    assert RiskSettings(exposure_basis="UNDERLYING").exposure_basis == "underlying"
    with pytest.raises(ValueError):
        RiskSettings(exposure_basis="foo")
