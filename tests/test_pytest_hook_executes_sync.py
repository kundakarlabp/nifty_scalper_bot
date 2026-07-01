from __future__ import annotations

pytest_plugins = ("pytester",)

_HOOK = """
import asyncio
import inspect
import pytest

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()

def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    loop = pyfuncitem.funcargs.get("event_loop")
    kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    if loop is None or not isinstance(loop, asyncio.AbstractEventLoop):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pyfuncitem.obj(**kwargs))
        finally:
            loop.close()
    else:
        loop.run_until_complete(pyfuncitem.obj(**kwargs))
    return True
"""


def test_pytest_hook_runs_sync_body(pytester):
    pytester.makeconftest(_HOOK)
    pytester.makepyfile(
        test_sync_body="""
        SENTINEL = False

        def test_sync_body_executes():
            global SENTINEL
            SENTINEL = True
            assert SENTINEL
        """
    )
    result = pytester.runpytest("-q")
    result.assert_outcomes(passed=1)


def test_pytest_hook_preserves_sync_assertion_failures(pytester):
    pytester.makeconftest(_HOOK)
    pytester.makepyfile(
        test_sync_failure="""
        def test_sync_failure_executes_and_fails():
            assert False, "sync body executed"
        """
    )
    result = pytester.runpytest("-q")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*sync body executed*"])


def test_pytest_hook_runs_async_body(pytester):
    pytester.makeconftest(_HOOK)
    pytester.makepyfile(
        test_async_body="""
        async def test_async_body_executes():
            assert True
        """
    )
    result = pytester.runpytest("-q")
    result.assert_outcomes(passed=1)
