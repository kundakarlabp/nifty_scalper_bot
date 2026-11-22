"""Legacy broker exits preserve product and never blindly retry uncertain calls."""

from __future__ import annotations

import logging
from datetime import time

from nifty_scalper_bot.execution.broker_rejects import BrokerReject
from nifty_scalper_bot.execution.exit_router import plan_and_send_exit

SYMBOL = "NFO:NIFTY26NOV25700CE"


def _route(now_time: time, product: str, reply):
    calls: list[tuple[str, str, str]] = []

    def submit(product_code: str, validity: str, variety: str):
        calls.append((product_code, validity, variety))
        return reply

    result = plan_and_send_exit(
        symbol=SYMBOL,
        quantity=65,
        product=product,
        now_time=now_time,
        submit=submit,
        logger=logging.getLogger(__name__),
    )
    return result, calls


def test_ambiguous_broker_timeout_cannot_send_second_sell() -> None:
    result, calls = _route(time(14, 30), "MIS", (False, None, "read timeout"))

    assert not result.ok
    assert result.reason is BrokerReject.UNKNOWN
    assert calls == [("MIS", "IOC", "regular")]


def test_throttle_or_cutoff_do_not_change_product_or_retry_without_reconciliation():
    for reply in (
        (False, None, "too many requests"),
        (False, None, "intraday orders (mis) are allowed only till 3:25"),
    ):
        result, calls = _route(time(15, 26), "MIS", reply)
        assert not result.ok
        assert calls == [("MIS", "IOC", "regular")]


def test_no_after_close_amo_for_intraday_position() -> None:
    result, calls = _route(time(15, 31), "MIS", (True, "fake", None))
    assert not calls
    assert result.reason is BrokerReject.MARKET_CLOSED


def test_no_after_close_amo_for_overday_position() -> None:
    result, calls = _route(time(15, 31), "NRML", (True, "fake", None))
    assert not calls
    assert result.reason is BrokerReject.MARKET_CLOSED


def test_success_without_order_id_does_not_claim_order_accepted() -> None:
    result, calls = _route(time(14, 30), "MIS", (True, None, None))
    assert not result.ok
    assert result.reason is BrokerReject.UNKNOWN
    assert len(calls) == 1


def test_normal_open_market_exit_keeps_broker_product() -> None:
    result, calls = _route(time(14, 30), "NRML", (True, "exit-65", None))

    assert result.ok
    assert result.order_id == "exit-65"
    assert calls == [("NRML", "IOC", "regular")]


def test_retry_max_is_compatibility_only_and_never_resubmits() -> None:
    calls: list[tuple[str, str, str]] = []

    def submit(product: str, validity: str, variety: str):
        calls.append((product, validity, variety))
        return False, None, "read timeout"

    for retry_max in (0, 1, 999):
        calls.clear()
        result = plan_and_send_exit(
            symbol=SYMBOL,
            quantity=65,
            product="MIS",
            now_time=time(14, 30),
            submit=submit,
            logger=logging.getLogger(__name__),
            retry_max=retry_max,
        )
        assert result.ok is False
        assert calls == [("MIS", "IOC", "regular")]
        assert len(result.attempts) == 1
        assert result.attempts[0].attempt == 1
