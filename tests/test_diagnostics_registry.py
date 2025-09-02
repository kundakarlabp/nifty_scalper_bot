from src.diagnostics import registry


def test_register_and_run_all():
    registry._registry.clear()

    @registry.register("foo")
    def _foo() -> registry.CheckResult:
        return registry.CheckResult(name="foo", ok=True, msg="ok", details={})

    res = registry.run("foo")
    assert res.ok and res.name == "foo"
    names = [r.name for r in registry.run_all()]
    assert names == ["foo"]
