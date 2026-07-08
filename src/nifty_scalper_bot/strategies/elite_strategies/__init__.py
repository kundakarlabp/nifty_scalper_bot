"""Public exports for elite strategies package."""

from __future__ import annotations

from .bb_squeeze import BBSqueezeStrategy
from .cpr_breakout import CPRBreakoutStrategy
from .gamma_scalping import GammaScalpingStrategy
from ..elite_tuesday_gamma_buyer import EliteTuesdayGammaBuyer
from .oi_max_pain import OIMaxPainStrategy
from .orb_pro import ORBProStrategy
from .order_flow import OrderFlowStrategy
from .order_flow_live_context_patch import apply_orderflow_live_context_patch
from .rsi_divergence import RSIDivergenceStrategy
from .smc_liquidity import SMCStrategy
from .straddle_theta import StraddleThetaStrategy
from .vwap_pro import VWAPProStrategy

apply_orderflow_live_context_patch(OrderFlowStrategy)

__all__ = [
    "BBSqueezeStrategy",
    "CPRBreakoutStrategy",
    "GammaScalpingStrategy",
    "EliteTuesdayGammaBuyer",
    "OIMaxPainStrategy",
    "ORBProStrategy",
    "OrderFlowStrategy",
    "RSIDivergenceStrategy",
    "SMCStrategy",
    "StraddleThetaStrategy",
    "VWAPProStrategy",
]
