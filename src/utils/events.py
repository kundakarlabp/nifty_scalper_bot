"""Utilities for loading and evaluating scheduled event guard windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Optional
import os
import yaml  # type: ignore


@dataclass
class GuardDefaults:
    """Default guard parameters applied when an event omits a field."""

    guard_before_min: int = 5
    guard_after_min: int = 5
    post_widen_spread_pct: float = 0.10
    block_trading: bool = True


@dataclass
class EventWindow:
    """Concrete guard window around a scheduled event."""

    name: str
    start: datetime
    end: datetime
    before_min: int
    after_min: int
    post_widen_spread_pct: float
    block_trading: bool

    def guard_start(self) -> datetime:
        return self.start - timedelta(minutes=self.before_min)

    def guard_end(self) -> datetime:
        return self.end + timedelta(minutes=self.after_min)

    def is_in_guard(self, now: datetime) -> bool:
        return self.guard_start() <= now <= self.guard_end()

    def is_in_event(self, now: datetime) -> bool:
        return self.start <= now <= self.end

    def is_in_post(self, now: datetime) -> bool:
        return self.end < now <= self.guard_end()


@dataclass
class EventCalendar:
    """Collection of event windows with helpers for lookups."""

    tz: ZoneInfo
    defaults: GuardDefaults
    events: List[EventWindow]
    version: int
    source_path: str
    mtime: float

    def active(self, now: datetime) -> List[EventWindow]:
        """Return event windows whose guard period covers ``now``."""
        return [e for e in self.events if e.is_in_guard(now)]

    def next_event(self, now: datetime) -> Optional[EventWindow]:
        """Return the next event whose guard starts after ``now``."""
        future = sorted(
            [e for e in self.events if e.guard_start() > now],
            key=lambda x: x.guard_start(),
        )
        return future[0] if future else None


def _parse_dt(s: str, tz: ZoneInfo) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _weekly_recurring(
    rule: str,
    start_local: str,
    end_local: str,
    tz: ZoneInfo,
    horizon_days: int = 14,
) -> List[tuple[datetime, datetime]]:
    """Expand a simple weekly rule into concrete datetime windows."""
    wd_map = {"WEEKLY_MON": 0, "WEEKLY_TUE": 1, "WEEKLY_WED": 2, "WEEKLY_THU": 3, "WEEKLY_FRI": 4}
    wd = wd_map.get(rule)
    if wd is None:
        return []
    hh, mm = map(int, start_local.split(":"))
    hh2, mm2 = map(int, end_local.split(":"))
    now = datetime.now(tz)
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    out: List[tuple[datetime, datetime]] = []
    for d in range(horizon_days):
        day = base + timedelta(days=d)
        if day.weekday() == wd:
            s = day.replace(hour=hh, minute=mm)
            e = day.replace(hour=hh2, minute=mm2)
            out.append((s, e))
    return out


def load_calendar(path: str) -> EventCalendar:
    """Load an :class:`EventCalendar` from a YAML file located at ``path``."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:  # pragma: no cover - yaml gives detailed msg
        raise ValueError(f"YAML parse error in {path}") from e
    tz = ZoneInfo(y.get("tz", "Asia/Kolkata"))
    defaults = GuardDefaults(
        guard_before_min=int(y.get("defaults", {}).get("guard_before_min", 5)),
        guard_after_min=int(y.get("defaults", {}).get("guard_after_min", 5)),
        post_widen_spread_pct=float(y.get("defaults", {}).get("post_widen_spread_pct", 0.10)),
        block_trading=bool(y.get("defaults", {}).get("block_trading", True)),
    )
    evs: List[EventWindow] = []
    for ev in y.get("events") or []:
        evs.append(
            EventWindow(
                name=str(ev["name"]),
                start=_parse_dt(ev["start"], tz),
                end=_parse_dt(ev["end"], tz),
                before_min=int(ev.get("guard_before_min", defaults.guard_before_min)),
                after_min=int(ev.get("guard_after_min", defaults.guard_after_min)),
                post_widen_spread_pct=float(ev.get("post_widen_spread_pct", defaults.post_widen_spread_pct)),
                block_trading=bool(ev.get("block_trading", defaults.block_trading)),
            )
        )
    for r in y.get("recurring") or []:
        rule = r.get("rule")
        s = r.get("start_local")
        end_local = r.get("end_local")
        for sdt, edt in _weekly_recurring(rule, s, end_local, tz):
            evs.append(
                EventWindow(
                    name=str(r.get("name", rule)),
                    start=sdt,
                    end=edt,
                    before_min=int(r.get("guard_before_min", defaults.guard_before_min)),
                    after_min=int(r.get("guard_after_min", defaults.guard_after_min)),
                    post_widen_spread_pct=float(
                        r.get("post_widen_spread_pct", defaults.post_widen_spread_pct)
                    ),
                    block_trading=bool(r.get("block_trading", defaults.block_trading)),
                )
            )
    cal = EventCalendar(
        tz=tz,
        defaults=defaults,
        events=sorted(evs, key=lambda x: x.start),
        version=int(y.get("version", 1)),
        source_path=path,
        mtime=os.path.getmtime(path),
    )
    return cal
