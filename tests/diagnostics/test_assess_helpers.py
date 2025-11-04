from nifty_scalper_bot.diagnostics.assess import CheckResult, assess_suite


def test_assess_suite_handles_success_and_exception() -> None:
    calls: list[str] = []

    def ok_check() -> tuple[bool, str, dict[str, object] | None]:
        calls.append("ok")
        return True, "passed", {"foo": "bar"}

    def failing_check() -> tuple[bool, str, dict[str, object] | None]:
        raise RuntimeError("boom")

    results = assess_suite([
        ("ok", ok_check),
        ("fail", failing_check),
    ])

    assert [result.name for result in results] == ["ok", "fail"]
    assert isinstance(results[0], CheckResult)
    assert results[0].ok is True
    assert results[0].detail == "passed"
    assert results[0].meta == {"foo": "bar"}
    assert results[1].ok is False
    assert results[1].detail.startswith("EXC: RuntimeError('boom')")
    assert results[1].meta == {}
    assert calls == ["ok"]
