#!/usr/bin/env python3
"""
Download all RCSB structures determined by X-ray diffraction
with resolution < 3 Å in mmCIF format, using rsync.

Workflow:
  1. Query RCSB Search API for matching PDB IDs
  2. rsync the full mmCIF mirror (~60 GB compressed)
  3. Remove files that don't match the query
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import requests

from src.config import DATA_DIR

# ── Configuration ────────────────────────────────────────────────
MIRROR_DIR = Path(f'{DATA_DIR}/mmCIF')          # local mirror directory
RSYNC_SOURCE = "rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/mmCIF/"

# ── Step 1: Query RCSB for matching PDB IDs ──────────────────────
def fetch_pdb_ids() -> set[str]:
    print("Querying RCSB Search API...")
    first_query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 3.0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }

    second_query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "ELECTRON MICROSCOPY",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 3.0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }

    resp = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=first_query)
    resp.raise_for_status()
    data = resp.json()
    first_ids = {hit["identifier"].lower() for hit in data["result_set"]}

    resp = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=second_query)
    resp.raise_for_status()
    data = resp.json()
    second_ids = {hit["identifier"].lower() for hit in data["result_set"]}

    ids = first_ids.union(second_ids)

    print(f"  Found {len(ids):,} matching structures")
    return ids

import gzip
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

RAW_DIR = Path(f"{DATA_DIR}/mmCIF/raw")
# EBI is closest to you (Belgium). RCSB alt: https://files.rcsb.org/download/{ID}.cif.gz
BASE = "https://files.rcsb.org/download"

_local = threading.local()   # requests.Session isn't thread-safe; one per thread

def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s

def _download_one(pdb_id: str) -> tuple[str, bool]:
    out = RAW_DIR / f"{pdb_id}.cif"
    if out.exists():
        return pdb_id, True
    mid2 = pdb_id[1:3]
    url = f"{BASE}/{pdb_id}.cif.gz"
    try:
        r = _session().get(url, timeout=60)
        r.raise_for_status()
        out.write_bytes(gzip.decompress(r.content))   # decompress on the fly
        return pdb_id, True
    except Exception:
        return pdb_id, False

def download_all(ids: set[str], workers: int = 16):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_download_one, i) for i in ids]
        for n, fut in enumerate(as_completed(futures), 1):
            pid, ok = fut.result()
            if not ok:
                failed.append(pid)
            if n % 1000 == 0:
                print(f"  {n:,}/{len(ids):,}  ({len(failed)} failed)")
    if failed:
        Path(f"{DATA_DIR}/metadata/failed_ids.txt").write_text("\n".join(failed))
        print(f"  {len(failed)} failed — retry them from failed_ids.txt")
    print(f"  Done. Files in {RAW_DIR}/")


def download_mmcif():
    ids = fetch_pdb_ids()

    id_file = Path(f'{DATA_DIR}/metadata/xray_lt3A_pdb_ids.txt')
    id_file.write_text("\n".join(sorted(ids)))
    print(f"  PDB IDs saved to {id_file}")

    download_all(ids, workers = 32)

    print(f"\nAll structures in: {MIRROR_DIR / 'raw'}/")
