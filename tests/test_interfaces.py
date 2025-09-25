from __future__ import annotations

"""Smoke tests for the public Protocol definitions."""

from typing import Protocol, runtime_checkable

from src import interfaces


def test_interfaces_module_exports_protocols() -> None:
    exported = {
        interfaces.DataSource,
        interfaces.Strategy,
        interfaces.Sizer,
        interfaces.Executor,
        interfaces.Notifier,
    }

    for proto in exported:
        assert isinstance(proto, type)
        # Protocols expose ``__mro_entries__`` which indicates they are runtime checkable.
        assert Protocol in proto.__mro__


def test_interfaces_runtime_checkable() -> None:
    @runtime_checkable
    class SupportsNotify(interfaces.Notifier, Protocol):
        """Concrete extension for runtime type checks."""

    class Implementation:
        def send(self, text: str) -> None:  # pragma: no cover - trivial
            self.last = text

    impl = Implementation()
    assert isinstance(impl, SupportsNotify)
