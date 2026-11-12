#!/usr/bin/env python3
"""Convert verified historical NSE NIFTY F&O EOD references into a private SQLite catalogue.

Read-only input. Rows are EOD observations, made available no earlier than the
next calendar day. They MUST NOT be used as same-day intraday contract identity,
order-book observations, or executable option prices. No external I/O or orders.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import tarfile
from datetime import date, timedelta

EXPECTED_FIRST = date(2024, 11, 6)
EXPECTED_LAST = date(2026, 11, 5)
MAX_FILE = 100_000_000
SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE sessions(
  day TEXT PRIMARY KEY, available_not_before TEXT NOT NULL,
  source_sha256 TEXT NOT NULL, source_url TEXT NOT NULL,
  source_archive TEXT NOT NULL, contracts INTEGER NOT NULL,
  option_contracts INTEGER NOT NULL, futures_contracts INTEGER NOT NULL,
  ex_post_only INTEGER NOT NULL CHECK(ex_post_only=1)
);
CREATE TABLE contract_eod(
  day TEXT NOT NULL REFERENCES sessions(day), instrument_id TEXT NOT NULL,
  instrument_name TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('FUT','CE','PE')),
  expiry TEXT NOT NULL, strike REAL, lot_size INTEGER,
  open REAL, high REAL, low REAL, close REAL, last REAL, settlement REAL,
  prev_close REAL, volume INTEGER, transactions INTEGER, open_interest INTEGER,
  turnover REAL, underlying REAL,
  source_row_index INTEGER NOT NULL, PRIMARY KEY(day,instrument_id,kind)
) WITHOUT ROWID;
CREATE INDEX contract_identity ON contract_eod(kind,expiry,strike,day);
CREATE INDEX contract_by_day ON contract_eod(day,kind,expiry);
CREATE TABLE daily_index(
  day TEXT PRIMARY KEY, available_not_before TEXT NOT NULL, open REAL NOT NULL,
  high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL,
  source TEXT NOT NULL, source_archive TEXT NOT NULL
);
"""

def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def check_sha(path:Path,sidecar:Path)->str:
    if not path.is_file() or not sidecar.is_file(): raise ValueError('source_archive_or_sha_missing')
    rows=sidecar.read_text().strip().split()
    if len(rows)!=2 or rows[1]!=path.name or len(rows[0])!=64: raise ValueError('sidecar_format_invalid')
    actual=sha_file(path)
    if actual!=rows[0]: raise ValueError('source_archive_sha_mismatch')
    return actual

def checked_tar(path:Path):
    """Validate inner manifest and return archive handle for streaming reads."""
    t=tarfile.open(path,'r:gz')
    members=t.getmembers()
    roots={item.name.split('/')[0] for item in members}
    if len(roots)!=1: t.close();raise ValueError('archive_root_invalid')
    root=next(iter(roots))
    for item in members:
        if item.issym() or item.islnk() or item.isdev() or item.size>MAX_FILE or '..' in Path(item.name).parts:
            t.close();raise ValueError('unsafe_archive_member')
    try:
        checksum_file=t.extractfile(root+'/SHA256SUMS')
        if checksum_file is None:raise ValueError('archive_manifest_missing')
        expected={}
        for line in checksum_file.read().decode('ascii').splitlines():
            expected_hash,relative=line.split('  ',1)
            if (relative in expected or relative.startswith('/') or '..' in Path(relative).parts
                or len(expected_hash)!=64 or any(c not in '0123456789abcdef' for c in expected_hash)):
                raise ValueError('archive_manifest_invalid')
            expected[relative]=expected_hash
        verified=0
        for item in members:
            if not item.isfile() or item.name==root+'/SHA256SUMS':continue
            relative=item.name[len(root)+1:]
            if relative not in expected:raise ValueError('archive_member_unlisted')
            stream=t.extractfile(item)
            if stream is None:raise ValueError('archive_member_unreadable')
            h=hashlib.sha256()
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=expected[relative]:raise ValueError('archive_member_digest_mismatch')
            verified+=1
        if verified!=len(expected):raise ValueError('archive_manifest_incomplete')
    except Exception:
        t.close();raise
    return t,root,verified

def number(row,key,*,integer=False):
    value=str(row.get(key,'')).strip()
    if not value:return None
    try:
        n=float(value)
    except ValueError as ex:raise ValueError('invalid_numeric_source_field') from ex
    if not math.isfinite(n):raise ValueError('nonfinite_source_field')
    if integer:
        if n<0 or not n.is_integer():raise ValueError('invalid_integer_source_field')
        return int(n)
    return n

def parse_catalog(raw:bytes,source_archive:str)->tuple[dict,list[tuple]]:
    payload=json.loads(gzip.decompress(raw))
    day=date.fromisoformat(payload['day'])
    if not EXPECTED_FIRST<=day<=EXPECTED_LAST:raise ValueError('date_outside_scope')
    nextday=day+timedelta(days=1)
    if payload['available_not_before']!=nextday.isoformat() or payload.get('ex_post_reference_only') is not True:
        raise ValueError('availability_or_evidence_incorrect')
    if len(payload['source_sha256'])!=64 or not payload['source_url'].startswith('https://'):
        raise ValueError('catalog_provenance_invalid')
    rows=payload['all_nifty_eod_rows']
    if not isinstance(rows,list) or not rows:raise ValueError('empty_catalog')
    options=[]; futures=0; sql=[]; ident=set()
    for i,row in enumerate(rows):
        if row.get('TckrSymb')!='NIFTY' or row.get('BizDt')!=day.isoformat():
            raise ValueError('catalog_identity_mismatch')
        typ=row['FinInstrmTp']
        if typ not in {'IDO','IDF'}:raise ValueError('catalog_type_invalid')
        expiry=date.fromisoformat(row['XpryDt'])
        if expiry<day:raise ValueError('catalog_expired_contract')
        kind='FUT' if typ=='IDF' else row['OptnTp']
        strike=number(row,'StrkPric') if typ=='IDO' else None
        if kind not in {'FUT','CE','PE'} or (typ=='IDO' and (strike is None or strike<=0)):
            raise ValueError('catalog_option_identity_invalid')
        if typ=='IDO': options.append((expiry.isoformat(),int(strike),kind))
        else:futures+=1
        token=row['FinInstrmId'].strip()
        key=(day.isoformat(),token,kind)
        if not token or key in ident:raise ValueError('catalog_duplicate_token')
        ident.add(key)
        lot=number(row,'NewBrdLotQty',integer=True)
        if lot is not None and lot<=0:raise ValueError('bad_reported_lot')
        sql.append((day.isoformat(),token,row['FinInstrmNm'],kind,expiry.isoformat(),strike,lot,
          number(row,'OpnPric'),number(row,'HghPric'),number(row,'LwPric'),
          number(row,'ClsPric'),number(row,'LastPric'),number(row,'SttlmPric'),
          number(row,'PrvsClsgPric'),number(row,'TtlTradgVol',integer=True),
          number(row,'TtlNbOfTxsExctd',integer=True),number(row,'OpnIntrst',integer=True),
          number(row,'TtlTrfVal'),number(row,'UndrlygPric'),
          i))
    declared=set(tuple(x) for x in payload['option_identities'])
    if set(options)!=declared or len(options)!=len(payload['option_identities']):
        raise ValueError('option_catalog_identity_list_mismatch')
    if set(payload['future_expiries']) != {r[4] for r in sql if r[3]=='FUT'}:
        raise ValueError('futures_expiry_list_mismatch')
    return ({'day':day.isoformat(),'nextday':nextday.isoformat(),'source_sha':payload['source_sha256'],
          'source_url':payload['source_url'],'source_archive':source_archive,'rows':len(rows),
          'options':len(options),'futures':futures},sql)

def build(main:Path,main_sha:Path,special:Path,special_sha:Path,output:Path)->dict:
    if output.exists():raise ValueError('output_must_be_new')
    checks={'main':check_sha(main,main_sha),'special':check_sha(special,special_sha)}
    original,root,verified=checked_tar(main)
    addon,aroot,verified_special=checked_tar(special)
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.tmp')
    if tmp.exists():raise ValueError('temporary_output_exists')
    counts={'sessions':0,'options':0,'futures':0,'daily_index':0,'main_members_verified':verified,
            'special_members_verified':verified_special,'source_sha256':checks,
            'research_only':True,'execution_grade_ready':False,'promotion_eligible':False,
            'market_data_tier':'EOD_EX_POST_NOT_EXECUTABLE','clock_verified':False}
    try:
        tmp.touch(mode=0o600,exist_ok=False)
        os.chmod(tmp,0o600)
        with sqlite3.connect(tmp) as conn:
            conn.executescript(SCHEMA)
            sources=[]
            for item in original.getmembers():
                if item.isfile() and item.name.startswith(root+'/catalog/') and item.name.endswith('.json.gz'):
                    sources.append((original,item,'main'))
            for item in addon.getmembers():
                if item.isfile() and item.name==aroot+'/2025-02-01.json.gz':
                    sources.append((addon,item,'special'))
            if len(sources)!=494:raise ValueError('expected_494_catalog_sessions')
            for archive,item,name in sorted(sources,key=lambda x:x[1].name):
                info,sql=parse_catalog(archive.extractfile(item).read(),name)
                conn.execute('INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,1)',
                  (info['day'],info['nextday'],info['source_sha'],info['source_url'],info['source_archive'],
                   info['rows'],info['options'],info['futures']))
                conn.executemany('INSERT INTO contract_eod VALUES ('+','.join(['?']*20)+')',sql)
                counts['sessions']+=1
                counts['options']+=info['options']
                counts['futures']+=info['futures']
            if len({v[0] for v in conn.execute('SELECT day FROM sessions')})!=494:
                raise ValueError('duplicate_or_missing_eod_sessions')
            with original.extractfile(root+'/index_daily/nifty_index_daily_indicators.csv') as f:
                lines=io.TextIOWrapper(f,encoding='utf8',newline='')
                for row in csv.DictReader(lines):
                    d=date.fromisoformat(row['date_ist']);
                    if not EXPECTED_FIRST<=d<=EXPECTED_LAST:raise ValueError('index_day_outside_scope')
                    vals=[float(row[k]) for k in ('open','high','low','close')]
                    if not all(math.isfinite(x) and x>0 for x in vals):raise ValueError('invalid_index_eod')
                    if not vals[2]<=min(vals[0],vals[3])<=max(vals[0],vals[3])<=vals[1]:
                        raise ValueError('index_eod_ohlc_conflict')
                    volume=float(row['reported_volume']) if row['reported_volume'] else None
                    conn.execute('INSERT INTO daily_index VALUES (?,?,?,?,?,?,?,?,?)',
                      (d.isoformat(),(d+timedelta(days=1)).isoformat(),*vals,volume,row['source'],'main'))
                    counts['daily_index']+=1
            if counts['sessions']!=494 or counts['options']!=829045 or counts['futures']!=1482 or counts['daily_index']!=493:
                raise ValueError('published_counts_reconciliation_failed')
            if conn.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
                raise ValueError('sqlite_integrity_error')
            conn.commit()
        os.replace(tmp,output)
        counts['sqlite_bytes']=output.stat().st_size
        counts['sqlite_sha256']=sha_file(output)
        return counts
    finally:
        original.close();addon.close()
        tmp.unlink(missing_ok=True)

def main_cli():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('main','main_sha','special','special_sha','output'):
        p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    x=p.parse_args()
    result=build(x.main,x.main_sha,x.special,x.special_sha,x.output)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main_cli()