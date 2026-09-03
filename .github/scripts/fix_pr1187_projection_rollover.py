from pathlib import Path

path = Path("src/nifty_scalper_bot/data/market_data_manager.py")
text = path.read_text(encoding="utf-8")
old = '''    @staticmethod
    def _projection_matches_canonical_slice(
        projection: list[tuple[Any, float, float, float, float, float]],
        canonical: list[tuple[Any, float, float, float, float, float]],
    ) -> bool:
        """Return True when projection rows are a contiguous canonical slice."""
        if not projection:
            return True
        if len(projection) > len(canonical):
            return False
        width = len(projection)
        return any(
            canonical[start : start + width] == projection
            for start in range(len(canonical) - width + 1)
        )
'''
new = '''    @staticmethod
    def _projection_matches_canonical_slice(
        projection: list[tuple[Any, float, float, float, float, float]],
        canonical: list[tuple[Any, float, float, float, float, float]],
    ) -> bool:
        """Return True when retained projection data agrees with canonical OHLC.

        A bounded CandleEngine window may legitimately roll older bars out between
        projection refreshes. In that case the old projection is no longer wholly
        contained in the new canonical window, but its retained suffix must equal
        the new canonical prefix. Any mismatch in the overlapping OHLCV rows still
        represents genuine projection divergence.
        """
        if not projection:
            return True
        if not canonical:
            return False

        if len(projection) <= len(canonical):
            width = len(projection)
            if any(
                canonical[start : start + width] == projection
                for start in range(len(canonical) - width + 1)
            ):
                return True

        max_overlap = min(len(projection), len(canonical))
        return any(
            projection[-overlap:] == canonical[:overlap]
            for overlap in range(max_overlap, 0, -1)
        )
'''
if old not in text:
    raise SystemExit("expected helper block not found; refusing broad edit")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
