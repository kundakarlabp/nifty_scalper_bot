"""Helpers for maintaining a persistent instrument universe cache."""

from __future__ import annotations

import csv
import os
import io
import sqlite3
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Iterable, Iterator

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS instruments(
      token INTEGER PRIMARY KEY,
      exchange TEXT NOT NULL,
      tradingsymbol TEXT NOT NULL,
      lot_size INTEGER NOT NULL,
      expiry TEXT,
      underlying TEXT,
      strike INTEGER,
      opt_type TEXT,
      updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_instruments_symbol ON instruments(tradingsymbol)",
    "CREATE INDEX IF NOT EXISTS idx_instruments_exchange ON instruments(exchange)",
)


@dataclass(slots=True)
class InstrumentUniverseStatus:
    """Mutable snapshot describing the cached instrument universe."""

    tokens: int = 0
    options: int = 0
    last_refresh: datetime | None = None
    last_source: str = "bootstrap"
    last_path: str | None = None

    def record_refresh(
        self,
        *,
        tokens: int,
        options: int,
        source: str,
        path: str | None,
        timestamp: datetime | None = None,
    ) -> None:
        """Update the refresh snapshot for reporting purposes.

        Args:
            tokens: Total resolver tokens available after the refresh.
            options: Number of option contracts tracked after the refresh.
            source: Human readable source label (e.g. ``"sqlite"``).
            path: Optional path hint describing the refresh artefact.
            timestamp: Override timestamp for ``last_refresh``; defaults to now.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered InstrumentUniverseStatus.record_refresh",
            extra={
                "event": "instrument_universe_record_refresh",
                "tokens": tokens,
                "options": options,
                "source": source,
            },
        )
        self.tokens = max(0, int(tokens))
        self.options = max(0, int(options))
        self.last_source = source
        self.last_path = path
        self.last_refresh = timestamp or datetime.now(timezone.utc)


def ensure_sqlite(path: str) -> sqlite3.Connection:
    """Return SQLite connection for the instrument cache, creating schema."""

    LOGGER.debug(
        "Entered ensure_sqlite",
        extra={"event": "instrument_cache_ensure_sqlite", "path": path},
    )
    try:
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
            timeout=60.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        with conn:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
        LOGGER.info(
            "Condition met: instrument_cache_ready",
            extra={
                "event": "instrument_cache_ready",
                "path": str(db_path),
            },
        )
        return conn
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in ensure_sqlite: %s",
            exc,
            extra={"event": "instrument_cache_ensure_error", "path": path},
            exc_info=exc,
        )
        raise


def _decode_bytes(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "iso-8859-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="ignore")


def _iter_csv_rows(text: str) -> Iterator[dict[str, Any]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return
    header_index = -1
    for idx, line in enumerate(lines):
        sample = next(csv.reader([line]))
        lowered = [token.strip().lower() for token in sample]
        if "instrument_token" in lowered or "instrumenttoken" in lowered:
            header_index = idx
            break
    if header_index < 0:
        return
    payload = "\n".join(lines[header_index:])
    stream = io.StringIO(payload)
    reader = csv.DictReader(stream)
    for row in reader:
        if not row:
            continue
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if key is None:
                continue
            safe_key = key.strip()
            if isinstance(value, str):
                normalized[safe_key] = value.strip()
            else:
                normalized[safe_key] = value
        if normalized:
            yield normalized


def parse_kite_csv(fp: IO[str] | IO[bytes]) -> Iterable[dict[str, Any]]:
    """Yield rows from a Zerodha instruments CSV or ZIP payload."""

    LOGGER.debug(
        "Entered parse_kite_csv",
        extra={"event": "instrument_cache_parse_enter"},
    )
    try:
        raw = fp.read()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in parse_kite_csv: %s",
            exc,
            extra={"event": "instrument_cache_parse_error"},
            exc_info=exc,
        )
        raise
    if isinstance(raw, str):
        payload = raw.encode("utf-8")
    else:
        payload = bytes(raw)
    buffer = io.BytesIO(payload)
    try:
        if zipfile.is_zipfile(buffer):
            with zipfile.ZipFile(buffer) as archive:
                for name in archive.namelist():
                    if not name.lower().endswith(".csv"):
                        continue
                    with archive.open(name) as member:
                        text = _decode_bytes(member.read())
                        yield from _iter_csv_rows(text)
                    return
            text = _decode_bytes(payload)
            yield from _iter_csv_rows(text)
            return
        text = _decode_bytes(payload)
        yield from _iter_csv_rows(text)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in parse_kite_csv: %s",
            exc,
            extra={"event": "instrument_cache_parse_failure"},
            exc_info=exc,
        )
        raise


def _derive_underlying(tradingsymbol: str) -> str:
    for index, char in enumerate(tradingsymbol):
        if char.isdigit():
            return tradingsymbol[:index].upper()
    return tradingsymbol.upper()


def _sanitize_opt_type(row: dict[str, Any], tradingsymbol: str) -> str | None:
    candidates = [
        row.get("instrument_type"),
        row.get("option_type"),
        row.get("instrumenttype"),
        row.get("opt_type"),
    ]
    for candidate in candidates:
        if candidate:
            token = str(candidate).strip().upper()
            if token in {"CE", "PE"}:
                return token
    suffix = tradingsymbol[-2:].upper()
    if suffix in {"CE", "PE"}:
        return suffix
    return None


def _allowed_instrument_prefixes(raw_filter: str) -> tuple[str, ...]:
    """Args: raw_filter; Returns: normalized prefixes; Raises: none."""

    tokens = [part.strip().upper() for part in str(raw_filter or "").split(",")]
    return tuple(token for token in tokens if token)


def upsert_instruments(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, Any]],
) -> dict[str, int]:
    """Upsert Zerodha instruments into the SQLite cache."""

    LOGGER.debug(
        "Entered upsert_instruments",
        extra={"event": "instrument_cache_upsert_enter"},
    )
    processed = 0
    stored = 0
    skipped = 0
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    settings = get_settings()
    instrument_settings = settings.instruments
    only_index_options = bool(
        getattr(instrument_settings, "sync_only_index_options", True)
    )
    allowed_prefixes = _allowed_instrument_prefixes(
        getattr(
            instrument_settings,
            "sync_instruments_filter",
            "NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY",
        )
    )
    try:
        upsert_data = []
        with conn:
            for raw_row in rows:
                processed += 1
                if not isinstance(raw_row, dict):
                    skipped += 1
                    continue
                tradingsymbol = (
                    str(raw_row.get("tradingsymbol") or raw_row.get("symbol") or "")
                    .strip()
                    .upper()
                )
                if not tradingsymbol:
                    skipped += 1
                    continue
                exchange = str(raw_row.get("exchange") or "").strip().upper()
                if exchange != "NFO":
                    skipped += 1
                    continue
                segment = str(raw_row.get("segment") or "").strip().upper()
                symbol_name = str(raw_row.get("name") or "").strip().upper()
                if only_index_options:
                    if segment != "NFO-OPT":
                        skipped += 1
                        continue
                    if not any(
                        symbol_name.startswith(prefix) for prefix in allowed_prefixes
                    ):
                        skipped += 1
                        continue
                if tradingsymbol.endswith("FUT"):
                    skipped += 1
                    continue
                opt_type = _sanitize_opt_type(raw_row, tradingsymbol)
                if opt_type not in {"CE", "PE"}:
                    skipped += 1
                    continue
                token_value = (
                    raw_row.get("instrument_token")
                    or raw_row.get("token")
                    or raw_row.get("instrumenttoken")
                )
                token_text = (
                    str(token_value).strip() if token_value not in {None, ""} else ""
                )
                try:
                    token_int = int(float(token_text))
                except (TypeError, ValueError):
                    skipped += 1
                    continue
                lot_size_raw = (
                    raw_row.get("lot_size")
                    or raw_row.get("lotsize")
                    or raw_row.get("lotSize")
                )
                lot_size_text = (
                    str(lot_size_raw).strip() if lot_size_raw not in {None, ""} else ""
                )
                try:
                    lot_size = max(1, int(float(lot_size_text)))
                except (TypeError, ValueError):
                    skipped += 1
                    continue
                expiry = raw_row.get("expiry")
                expiry_str = str(expiry).strip() if expiry else None
                strike_raw = raw_row.get("strike") or raw_row.get("strike_price")
                strike_value: int | None = None
                if strike_raw not in {None, ""}:
                    strike_text = str(strike_raw).strip()
                    with suppress(Exception):
                        strike_value = int(round(float(strike_text)))
                underlying = raw_row.get("name")
                underlying_str = (
                    str(underlying).strip().upper()
                    if underlying
                    else _derive_underlying(tradingsymbol)
                )
                upsert_data.append(
                    (
                        token_int,
                        exchange,
                        tradingsymbol,
                        lot_size,
                        expiry_str,
                        underlying_str,
                        strike_value,
                        opt_type,
                        timestamp,
                    )
                )

            if upsert_data:
                conn.executemany(
                    """
                    INSERT INTO instruments(
                        token,
                        exchange,
                        tradingsymbol,
                        lot_size,
                        expiry,
                        underlying,
                        strike,
                        opt_type,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(token) DO UPDATE SET
                        exchange=excluded.exchange,
                        tradingsymbol=excluded.tradingsymbol,
                        lot_size=excluded.lot_size,
                        expiry=excluded.expiry,
                        underlying=excluded.underlying,
                        strike=excluded.strike,
                        opt_type=excluded.opt_type,
                        updated_at=excluded.updated_at
                    """,
                    upsert_data,
                )
                stored = len(upsert_data)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in upsert_instruments: %s",
            exc,
            extra={"event": "instrument_cache_upsert_error"},
            exc_info=exc,
        )
        raise
    options = stored
    METRICS.record_resolver_learn(
        count=stored,
        label="bootstrap" if stored and processed == stored else "refresh",
    )
    return {
        "processed": processed,
        "stored": stored,
        "skipped": skipped,
        "options": options,
    }


def load_rows_for_resolver(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return broker-style rows suitable for resolver warm-up."""

    LOGGER.debug(
        "Entered load_rows_for_resolver",
        extra={"event": "instrument_cache_load_enter"},
    )
    try:
        cursor = conn.execute("""
            SELECT
                token,
                tradingsymbol,
                exchange,
                lot_size,
                expiry,
                strike,
                opt_type
            FROM instruments
            ORDER BY token ASC
            """)
        rows: list[dict[str, Any]] = []
        for entry in cursor.fetchall():
            token_int = int(entry["token"])
            broker_row: dict[str, Any] = {
                "instrument_token": token_int,
                "tradingsymbol": entry["tradingsymbol"],
                "exchange": entry["exchange"],
                "lot_size": int(entry["lot_size"]),
            }
            if entry["expiry"]:
                broker_row["expiry"] = entry["expiry"]
            if entry["strike"] is not None:
                broker_row["strike"] = entry["strike"]
            if entry["opt_type"]:
                broker_row["instrument_type"] = entry["opt_type"]
            rows.append(broker_row)
        return rows
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in load_rows_for_resolver: %s",
            exc,
            extra={"event": "instrument_cache_load_error"},
            exc_info=exc,
        )
        raise


def refresh_from_csv(
    conn: sqlite3.Connection,
    csv_path: str,
) -> dict[str, Any]:
    """Refresh the instrument cache from a Zerodha CSV dump."""

    LOGGER.debug(
        "Entered refresh_from_csv",
        extra={"event": "instrument_cache_refresh_enter", "path": csv_path},
    )
    file_path = Path(csv_path)
    if not file_path.exists():
        LOGGER.info(
            "Condition met: instrument_cache_refresh_missing",
            extra={"event": "instrument_cache_refresh_missing", "path": csv_path},
        )
        return {"processed": 0, "stored": 0, "skipped": 0, "options": 0}
    try:
        with file_path.open("rb") as handle:
            rows = list(parse_kite_csv(handle))
        metrics: dict[str, Any] = dict(upsert_instruments(conn, rows))
        metrics["path"] = str(file_path)
        metrics["source"] = "csv"
        LOGGER.info(
            "Condition met: instrument_cache_refresh_success",
            extra={
                "event": "instrument_cache_refresh_success",
                "path": str(file_path),
                "stored": metrics.get("stored"),
                "processed": metrics.get("processed"),
            },
        )
        return metrics
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in refresh_from_csv: %s",
            exc,
            extra={"event": "instrument_cache_refresh_error", "path": csv_path},
            exc_info=exc,
        )
        raise


def write_instrument_rows_to_csv(
    rows: Iterable[dict[str, Any]],
    csv_path: str,
) -> int:
    """Write broker rows to CSV.

    Args: rows,csv_path.
    Returns: Number of rows written.
    Raises: Exception on I/O failure.
    """

    LOGGER.debug(
        "Entered write_instrument_rows_to_csv",
        extra={"event": "instrument_csv_write_enter", "path": csv_path},
    )
    serializable_rows = [dict(row) for row in rows if isinstance(row, dict)]
    if not serializable_rows:
        LOGGER.info(
            "Condition met: instrument_csv_write_skipped",
            extra={"event": "instrument_csv_write_skipped", "path": csv_path},
        )
        return 0
    try:
        field_names: list[str] = []
        seen: set[str] = set()
        for row in serializable_rows:
            for key in row:
                clean_key = str(key).strip()
                if not clean_key or clean_key in seen:
                    continue
                seen.add(clean_key)
                field_names.append(clean_key)
        if "instrument_token" not in seen and "instrumenttoken" in seen:
            field_names = [
                "instrument_token" if name == "instrumenttoken" else name
                for name in field_names
            ]
        if "instrument_token" not in field_names:
            field_names.insert(0, "instrument_token")
        target_path = Path(csv_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = target_path.with_suffix(".tmp")
        with tmp_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=field_names, extrasaction="ignore"
            )
            writer.writeheader()
            for raw_row in serializable_rows:
                row = dict(raw_row)
                if row.get("instrument_token") in {None, ""} and row.get(
                    "instrumenttoken"
                ) not in {None, ""}:
                    row["instrument_token"] = row.get("instrumenttoken")
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_file.replace(target_path)
        LOGGER.info(
            "Condition met: instrument_csv_write_success",
            extra={
                "event": "instrument_csv_write_success",
                "path": str(target_path),
                "written": len(serializable_rows),
            },
        )
        return len(serializable_rows)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in write_instrument_rows_to_csv: %s",
            exc,
            extra={"event": "instrument_csv_write_error", "path": csv_path},
            exc_info=exc,
        )
        raise


def sync_instrument_csv_from_broker(
    broker_client: Any,
    csv_path: str,
    *,
    exchange: str = "NFO",
) -> dict[str, Any]:
    """Sync broker instruments into CSV.

    Args: broker_client,csv_path,exchange.
    Returns: Sync metrics.
    Raises: Exception when broker retrieval fails.
    """

    LOGGER.debug(
        "Entered sync_instrument_csv_from_broker",
        extra={
            "event": "instrument_csv_sync_enter",
            "path": csv_path,
            "exchange": exchange,
        },
    )
    normalized_exchange = str(exchange or "NFO").strip().upper()
    try:
        rows: list[dict[str, Any]] = []
        load_fn = getattr(broker_client, "load_instruments", None)
        if callable(load_fn):
            payload = load_fn(normalized_exchange)
            if isinstance(payload, list):
                rows = [row for row in payload if isinstance(row, dict)]
        if not rows:
            list_fn = getattr(broker_client, "list_instruments", None)
            if callable(list_fn):
                payload = list_fn()
                if isinstance(payload, list):
                    rows = [
                        row
                        for row in payload
                        if isinstance(row, dict)
                        and str(row.get("exchange") or "").strip().upper()
                        == normalized_exchange
                    ]
        written = write_instrument_rows_to_csv(rows, csv_path)
        LOGGER.info(
            "Condition met: instrument_csv_sync_success",
            extra={
                "event": "instrument_csv_sync_success",
                "exchange": normalized_exchange,
                "path": csv_path,
                "fetched": len(rows),
                "written": written,
            },
        )
        return {
            "exchange": normalized_exchange,
            "fetched": len(rows),
            "written": written,
            "path": csv_path,
            "source": "broker",
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in sync_instrument_csv_from_broker: %s",
            exc,
            extra={
                "event": "instrument_csv_sync_error",
                "path": csv_path,
                "exchange": normalized_exchange,
            },
            exc_info=exc,
        )
        raise


__all__ = [
    "InstrumentUniverseStatus",
    "ensure_sqlite",
    "parse_kite_csv",
    "upsert_instruments",
    "load_rows_for_resolver",
    "refresh_from_csv",
    "sync_instrument_csv_from_broker",
    "write_instrument_rows_to_csv",
]
