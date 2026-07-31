"""Utilities to ensure live and backtest pipelines remain identical."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from nifty_scalper_bot.backtest.replay import ReplayHarness, ReplayResult
from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class PipelineParityResult:
    """Outcome of a parity run comparing live and backtest pipelines."""

    live: ReplayResult
    backtest: ReplayResult
    diff: str


class SinglePipelineParity:
    """Run the same component pipeline for live and backtest replay."""

    def __init__(
        self,
        *,
        runner_factory: Callable[[], Any],
        reference_runner_factory: Callable[[], Any] | None = None,
        paper_factory: Callable[[], PaperFillEngine],
        option_symbol: str,
        index_symbol: str | None = None,
    ) -> None:
        """Initialise the parity harness using shared factories.

        Args:
            runner_factory: Callable returning a fresh strategy runner instance.
            paper_factory: Callable returning a fresh paper fill engine instance.
            option_symbol: Primary option symbol processed by the harness.
            index_symbol: Optional index symbol forwarded to the harness.

        Returns:
            None.

        Raises:
            None.
        """

        self._runner_factory = runner_factory
        self._reference_runner_factory = reference_runner_factory
        self._paper_factory = paper_factory
        self._option_symbol = option_symbol
        self._index_symbol = index_symbol
        self._logger = LOGGER

    def run_frame(self, frame: pd.DataFrame) -> PipelineParityResult:
        """Execute both live and backtest passes over *frame* and compare results.

        Args:
            frame: Market data frame containing replay ticks.

        Returns:
            PipelineParityResult describing live/backtest outcomes and diff.

        Raises:
            Exception: Propagates errors emitted by underlying components.
        """

        self._logger.debug(
            "Entered SinglePipelineParity.run_frame",
            extra={"event": "single_pipeline_parity_run_enter"},
        )
        if self._reference_runner_factory is None:
            raise ValueError(
                "reference_runner_factory is required; comparing one replay "
                "pipeline with itself does not prove parity"
            )
        try:
            live_runner = self._runner_factory()
            live_paper = self._paper_factory()
            live_harness = ReplayHarness(
                live_runner,
                live_paper,
                option_symbol=self._option_symbol,
                index_symbol=self._index_symbol,
            )
            live_result = live_harness.run_dataframe(frame)

            back_runner = self._reference_runner_factory()
            back_paper = self._paper_factory()
            back_harness = ReplayHarness(
                back_runner,
                back_paper,
                option_symbol=self._option_symbol,
                index_symbol=self._index_symbol,
            )
            back_result = back_harness.run_dataframe(frame)
        except Exception as exc:
            self._logger.error(
                "Failure in SinglePipelineParity.run_frame: %s",
                exc,
                extra={"event": "single_pipeline_parity_run_error"},
            )
            raise
        diff = self._diff_trades(live_result.trades, back_result.trades)
        if diff:
            self._logger.error(
                "Condition met: pipeline_parity_mismatch",
                extra={"event": "single_pipeline_parity_diff", "diff": diff},
            )
        else:
            self._logger.info(
                "Condition met: pipeline_parity_match",
                extra={"event": "single_pipeline_parity_match"},
            )
        return PipelineParityResult(live=live_result, backtest=back_result, diff=diff)

    @staticmethod
    def _diff_trades(live: list[dict[str, Any]], back: list[dict[str, Any]]) -> str:
        """Return unified diff for two trade logs, or ``""`` if identical.

        Args:
            live: Trade log captured from the live-style replay.
            back: Trade log captured from the backtest replay.

        Returns:
            Diff string describing mismatches, or an empty string on parity.

        Raises:
            None.
        """

        live_json = json.dumps(live, sort_keys=True, default=str, ensure_ascii=False)
        back_json = json.dumps(back, sort_keys=True, default=str, ensure_ascii=False)
        if live_json == back_json:
            return ""
        diff_lines = difflib.unified_diff(
            live_json.splitlines(),
            back_json.splitlines(),
            fromfile="live",
            tofile="back",
        )
        return "\n".join(diff_lines)


__all__ = ["PipelineParityResult", "SinglePipelineParity"]
