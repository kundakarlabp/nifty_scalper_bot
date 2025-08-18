Use in a test when you need specific env driven settings:

    def test_uses_test_key():
        # temp_env is injected by conftest.py (no import needed)
        with temp_env(ZERODHA_API_KEY="test_key"):
            import importlib, src.config as appconf
            importlib.reload(appconf)
            assert getattr(appconf, "settings", None)  # whatever your module provides
            # assert appconf.settings.api_key == "test_key"  # adapt to your shape

