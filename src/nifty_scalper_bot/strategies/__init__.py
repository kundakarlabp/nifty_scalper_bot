from .elite_strategies import *  # noqa: F401,F403

try:
    from .signal_identity_patch import apply_patches as _apply_signal_identity_patches

    _apply_signal_identity_patches()
except Exception:
    pass

try:
    from .runtime_context_contract import install_indicator_runtime_context_contract

    install_indicator_runtime_context_contract()
except Exception:
    pass

try:
    from .elite_strategies import __all__ as _elite_all  # type: ignore

    __all__ = list(_elite_all)
except Exception:
    __all__ = []
