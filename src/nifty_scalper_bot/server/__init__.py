"""HTTP routes supporting diagnostics and integrations."""

from .selftest import router as selftest_router

__all__ = ["selftest_router"]
