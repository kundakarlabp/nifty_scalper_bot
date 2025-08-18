import os
import importlib
import contextlib
import pytest

# Ensure ENV_FILE is set for every test run
TEST_ENV = os.path.join(os.getcwd(), "tests", ".env.test")
os.environ.setdefault("ENV_FILE", TEST_ENV)
os.environ.setdefault("TZ", "Asia/Kolkata")

@contextlib.contextmanager
def temp_env(**overrides):
    """Temporarily set env vars, reload config, then restore."""
    old = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None and k in os.environ:
                del os.environ[k]
            elif v is not None:
                os.environ[k] = str(v)
        # reload config modules to re-read env
        try:
            import src.utils.config as uconf
            importlib.reload(uconf)
        except Exception:
            uconf = None
        import src.config as appconf
        importlib.reload(appconf)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # reload back
        try:
            import src.utils.config as uconf
            importlib.reload(uconf)
        except Exception:
            pass
        import src.config as appconf
        importlib.reload(appconf)

@pytest.fixture(autouse=True)
def ensure_test_env_and_reload(monkeypatch):
    """Autouse: before every test, enforce ENV_FILE + TZ and reload config."""
    monkeypatch.setenv("ENV_FILE", TEST_ENV)
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    # reload config so BaseSettings re-parses env on each test
    try:
        import src.utils.config as uconf  # optional helper you added earlier
        importlib.reload(uconf)
    except Exception:
        pass
    import src.config as appconf
    importlib.reload(appconf)
    # expose helper for tests that need scoped overrides
    import builtins
    builtins.temp_env = temp_env
    yield
