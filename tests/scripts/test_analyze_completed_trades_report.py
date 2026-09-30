from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reporting" / "analyze_completed_trades.py"
spec = importlib.util.spec_from_file_location("analyze_completed_trades", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_build_analysis_emits_post_cost_attribution_groups() -> None:
    quality = {"alpha_score": 8.0, "final_score": 8.1, "strategy_score": 7.5}
    outcome = {
        "cost_source": "broker_virtual_contract_note",
        "effective_costs": {"total": 20.0},
        "signal_quality": quality,
        "strategy_key": "vwap_pro",
        "strategy_role": "trigger",
        "signal_family": "directional_trigger",
        "setup_name": "continuation_pullback",
        "regime": "TREND",
        "approval_path": "single_trigger_context_confirmed",
        "score_contract_version": 1,
        "score_lineage": {
            "raw_setup_score": 7.5,
            "regime_weight": 1.0,
            "regime_adjusted_setup_score": 7.5,
            "context_confirmation_bonus": 0.0,
            "context_veto_penalty": 0.0,
            "manager_reference_score": 7.5,
            "manager_reference_threshold": 7.0,
            "manager_reference_pass": True,
            "final_numeric_gate_owner": "runner_final_execution_score",
        },
        "confirming_trigger_strategies": ["VWAPPro"],
        "context_confirmation_strategies": ["OrderFlow"],
    }
    row = {
        "trade_id": "t1", "closed_at": 1.0, "strategy": "VWAPPro",
        "gross_pnl": 100.0, "estimated_costs": 20.0, "net_pnl": 80.0,
        "ledger_complete": True, "state": "CLOSED", "exit_reason": "TARGET",
        "outcome": outcome,
    }

    report = module.build_analysis([row], block_size=20, components=("VWAPPro",))

    assert report["attribution"]["ready"] is True
    assert report["attribution"]["groups"] == [{
        "regime": "TREND",
        "setup_name": "continuation_pullback",
        "confirmation_type": "single_trigger_context_confirmed",
        "summary": {
            "trade_count": 1, "gross_pnl": 100.0, "estimated_costs": 20.0,
            "effective_costs": 20.0, "broker_cost_trade_count": 1,
            "net_pnl": 80.0, "expectancy": 80.0, "win_rate": 1.0,
            "average_win": 80.0, "average_loss": 0.0, "profit_factor": None,
            "max_drawdown": 0.0,
        },
    }]
