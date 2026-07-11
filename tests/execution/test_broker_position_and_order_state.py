import pytest

from nifty_scalper_bot.execution.broker_position_evidence import (
    BrokerPositionState,
    evidence_from_positions,
)
from nifty_scalper_bot.execution.order_state import (
    DomainOrderState,
    map_broker_order_status,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PUT ORDER REQ RECEIVED", DomainOrderState.SUBMISSION_PENDING),
        ("VALIDATION PENDING", DomainOrderState.SUBMISSION_PENDING),
        ("OPEN PENDING", DomainOrderState.SUBMISSION_PENDING),
        ("OPEN", DomainOrderState.OPEN),
        ("TRIGGER PENDING", DomainOrderState.TRIGGER_PENDING),
        ("MODIFY VALIDATION PENDING", DomainOrderState.OPEN),
        ("MODIFY PENDING", DomainOrderState.OPEN),
        ("CANCEL PENDING", DomainOrderState.CANCEL_PENDING),
        ("COMPLETE", DomainOrderState.FILLED),
        ("REJECTED", DomainOrderState.REJECTED),
        ("CANCELLED", DomainOrderState.CANCELLED),
        ("SURPRISE", DomainOrderState.UNKNOWN),
        ("", DomainOrderState.UNKNOWN),
    ],
)
def test_order_status_mapping(raw, expected):
    evidence = map_broker_order_status(raw)
    assert evidence.state is expected
    assert evidence.raw_status == raw.upper()


def test_empty_valid_positions_response_is_flat():
    evidence = evidence_from_positions("NFO:NIFTY2670124000CE", [])
    assert evidence.state is BrokerPositionState.FLAT_CONFIRMED
    assert evidence.net_quantity == 0


def test_no_prefix_kite_symbol_matches_nfo_prefixed_local_symbol():
    evidence = evidence_from_positions(
        "NFO:NIFTY2670124000CE", [{"tradingsymbol": "NIFTY2670124000CE", "quantity": 0}]
    )
    assert evidence.state is BrokerPositionState.FLAT_CONFIRMED


def test_partial_position_is_non_flat():
    evidence = evidence_from_positions(
        "NFO:NIFTY2670124000CE",
        [{"tradingsymbol": "NIFTY2670124000CE", "quantity": 25}],
    )
    assert evidence.state is BrokerPositionState.NON_FLAT_CONFIRMED
    assert evidence.net_quantity == 25


def test_multiple_position_rows_are_net_aggregated():
    rows = [
        {"tradingsymbol": "NIFTY2670124000CE", "quantity": 65, "product": "MIS"},
        {"tradingsymbol": "NIFTY2670124000CE", "quantity": -65, "product": "NRML"},
    ]
    evidence = evidence_from_positions("NFO:NIFTY2670124000CE", rows)
    assert evidence.state is BrokerPositionState.FLAT_CONFIRMED
    assert evidence.net_quantity == 0


def test_negative_quantity_is_non_flat():
    evidence = evidence_from_positions(
        "NFO:NIFTY2670124000CE",
        [{"tradingsymbol": "NIFTY2670124000CE", "quantity": -65}],
    )
    assert evidence.state is BrokerPositionState.NON_FLAT_CONFIRMED
    assert evidence.net_quantity == -65


def test_malformed_positions_are_unknown_not_flat():
    evidence = evidence_from_positions(
        "NFO:NIFTY2670124000CE",
        [{"tradingsymbol": "NIFTY2670124000CE", "quantity": "bad"}],
    )
    assert evidence.state is BrokerPositionState.UNKNOWN
    assert evidence.net_quantity is None


def test_live_exit_patch_apply_is_deprecated_no_method_reassignment():
    from nifty_scalper_bot.execution import live_exit_reconciliation_patch
    from nifty_scalper_bot.execution.ownership import BoundBracketManager

    before = BoundBracketManager._reconcile_exit_state
    with pytest.warns(DeprecationWarning):
        live_exit_reconciliation_patch.apply_patches()
    assert BoundBracketManager._reconcile_exit_state is before
