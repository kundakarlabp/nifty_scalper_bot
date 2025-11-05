#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import sys
import time
import zipfile

from dateutil.relativedelta import relativedelta
import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/",
}
URL_PATTERNS = [
    "https://archives.nseindia.com/content/historical/DERIVATIVES/{YYYY}/{MON}/fo{DDMONYYYY}bhav.csv.zip",
    "https://www.nseindia.com/content/historical/DERIVATIVES/{YYYY}/{MON}/fo{DDMONYYYY}bhav.csv.zip",
]


def last_thursday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1) + relativedelta(months=1, days=-1)
    while d.weekday() != 3:
        d -= dt.timedelta(days=1)
    return d


def classify(expiry: dt.date) -> str:
    return "MONTHLY" if expiry == last_thursday(expiry.year, expiry.month) else "WEEKLY"


def ddrange(days: int, end_date: dt.date | None = None):
    end = end_date or dt.date.today()
    for i in range(days):
        yield end - dt.timedelta(days=i)


def url_for(d: dt.date):
    mon = d.strftime("%b").upper()
    token = d.strftime("%d%b%Y").upper()
    yy = d.strftime("%Y")
    return [p.format(YYYY=yy, MON=mon, DDMONYYYY=token) for p in URL_PATTERNS]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def fetch_zip_to(
    path: str,
    urls: list[str],
    session: requests.Session,
    verbose: bool = False,
) -> bool:
    if os.path.exists(path):
        if verbose:
            print(f"[cache] exists: {path}")
        return True
    for u in urls:
        for attempt in range(1, 4):
            try:
                r = session.get(u, headers=HEADERS, timeout=30)
                if r.status_code == 200 and r.content and r.content[:2] == b"PK":
                    with open(path, "wb") as f:
                        f.write(r.content)
                    if verbose:
                        print(f"[ok] saved: {path}")
                    return True
            except requests.RequestException:
                pass
            time.sleep(2 * attempt)
    if verbose:
        print(f"[miss] no zip found for: {os.path.basename(path)}")
    return False


def parse_zip_to_df(zpath: str) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            inner = [n for n in zf.namelist() if n.lower().endswith("bhav.csv")]
            if not inner:
                return None
            with zf.open(inner[0]) as f:
                df = pd.read_csv(f)
    except Exception:
        return None
    # Filter NIFTY OPTIDX
    df = df[(df["INSTRUMENT"] == "OPTIDX") & (df["SYMBOL"] == "NIFTY")]
    if df.empty:
        return None
    for col in ("EXPIRY_DT", "TIMESTAMP"):
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                format="%d-%b-%Y",
                errors="coerce",
            ).dt.date
    df["EXPIRY_CLASS"] = df["EXPIRY_DT"].apply(
        lambda d: classify(d) if pd.notnull(d) else None
    )
    keep = [
        "TIMESTAMP",
        "INSTRUMENT",
        "SYMBOL",
        "EXPIRY_DT",
        "EXPIRY_CLASS",
        "STRIKE_PR",
        "OPTION_TYP",
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "SETTLE_PR",
        "CONTRACTS",
        "VAL_INLAKH",
        "OPEN_INT",
        "CHG_IN_OI",
    ]
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Sync last N days of NIFTY options (OPTIDX) from NSE F&O bhavcopies."
        )
    )
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--outdir", type=str, default="data")
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    cache_root = os.path.join(outdir, "cache", "fo")
    ensure_dir(cache_root)
    ensure_dir(outdir)

    session = requests.Session()
    zips = []
    for d in ddrange(args.days):
        if d.weekday() == 6:  # Sunday
            continue
        mon = d.strftime("%b").upper()
        yy = d.strftime("%Y")
        cache_dir = os.path.join(cache_root, yy, mon)
        ensure_dir(cache_dir)
        fname = f"fo{d.strftime('%d%b%Y').upper()}bhav.csv.zip"
        zpath = os.path.join(cache_dir, fname)
        urls = url_for(d)
        ok = fetch_zip_to(zpath, urls, session, verbose=args.verbose)
        if ok:
            zips.append(zpath)

    if not zips and not args.force_rebuild:
        print("[warn] no zips fetched/found; nothing to build", file=sys.stderr)
        sys.exit(2)

    frames = []
    for z in sorted(zips):
        df = parse_zip_to_df(z)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames and not args.force_rebuild:
        print("[warn] no NIFTY OPTIDX rows found; exiting", file=sys.stderr)
        sys.exit(3)

    all_path = os.path.join(outdir, "nifty_options_all.csv")
    weekly_path = os.path.join(outdir, "nifty_options_weekly.csv")
    monthly_path = os.path.join(outdir, "nifty_options_monthly.csv")

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df = all_df.sort_values(
            ["TIMESTAMP", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP"]
        )
        all_df.to_csv(all_path, index=False)
        all_df[all_df["EXPIRY_CLASS"] == "WEEKLY"].to_csv(weekly_path, index=False)
        all_df[all_df["EXPIRY_CLASS"] == "MONTHLY"].to_csv(
            monthly_path,
            index=False,
        )
        print(f"[ok] total rows: {len(all_df):,}")
    else:
        # If --force-rebuild with no frames, create empty CSVs with headers
        hdr = [
            "TIMESTAMP",
            "INSTRUMENT",
            "SYMBOL",
            "EXPIRY_DT",
            "EXPIRY_CLASS",
            "STRIKE_PR",
            "OPTION_TYP",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "SETTLE_PR",
            "CONTRACTS",
            "VAL_INLAKH",
            "OPEN_INT",
            "CHG_IN_OI",
        ]
        pd.DataFrame(columns=hdr).to_csv(all_path, index=False)
        pd.DataFrame(columns=hdr).to_csv(weekly_path, index=False)
        pd.DataFrame(columns=hdr).to_csv(monthly_path, index=False)
        print("[warn] built empty CSVs (force-rebuild)")

    print(f"[ok] saved: {all_path}")
    print(f"[ok] saved: {weekly_path}")
    print(f"[ok] saved: {monthly_path}")


if __name__ == "__main__":
    main()
