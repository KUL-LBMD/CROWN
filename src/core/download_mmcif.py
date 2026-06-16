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
    query = {
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

    resp = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
    resp.raise_for_status()
    data = resp.json()
    ids = {hit["identifier"].lower() for hit in data["result_set"]}
    print(f"  Found {len(ids):,} matching structures")
    return ids


# ── Step 2: rsync the full mmCIF archive ─────────────────────────
def rsync_mirror():
    print(f"\nSyncing full mmCIF mirror to {MIRROR_DIR}/")
    print("  (This is ~60 GB compressed — grab a coffee.)\n")
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync",
        "-rlpt",        # recursive, symlinks, permissions, timestamps
        "-v",            # verbose
        "-z",            # compress during transfer
        "--delete",      # remove local files no longer on server
        "--info=progress2",
        RSYNC_SOURCE,
        str(MIRROR_DIR) + "/",
    ]
    subprocess.run(cmd, check=True)
    print("  rsync complete.")


# ── Step 3: Filter — keep only matching structures ───────────────
def filter_mirror(keep_ids: set[str]):
    """
    mmCIF mirror layout:
        mmCIF/<mid2>/<pdb_id>.cif.gz
    e.g.  mmCIF/ab/1abc.cif.gz

    The middle-two-char subdirectory is pdb_id[1:3].
    """
    all_cifs = sorted(MIRROR_DIR.rglob("*.cif.gz"))
    print(f"\nTotal files in mirror: {len(all_cifs):,}")

    to_keep = []
    to_remove = []
    for path in all_cifs:
        pdb_id = path.stem.replace(".cif", "").lower()  # strip .cif.gz
        if pdb_id in keep_ids:
            to_keep.append(path)
        else:
            to_remove.append(path)

    print(f"  Keeping:  {len(to_keep):,}")
    print(f"  Removing: {len(to_remove):,}")

    for p in to_remove:
        p.unlink()
    # clean up empty subdirectories
    for subdir in sorted(MIRROR_DIR.iterdir(), reverse=True):
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()

def decompress_all():
    """Decompress all .cif.gz files in place and remove the originals."""
    import gzip
    import shutil

    gz_files = sorted(MIRROR_DIR.rglob("*.cif.gz"))
    print(f"\nDecompressing {len(gz_files):,} files...")

    for i, gz_path in enumerate(gz_files):
        out_path = gz_path.with_suffix("")  # strips .gz → keeps .cif
        try:
            with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
        except Exception as e:
            print(f"  Failed: {gz_path.name}: {e}")

        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1:,}/{len(gz_files):,} done")

    print("  Decompression complete.")

def flatten_mirror():
    """Move all .cif files into a single flat directory and remove empty subdirs."""
    flat_dir = MIRROR_DIR / "all"
    flat_dir.mkdir(exist_ok=True)

    cif_files = sorted(MIRROR_DIR.rglob("*.cif"))
    # Exclude files already in the target dir
    cif_files = [f for f in cif_files if f.parent != flat_dir]
    print(f"\nFlattening {len(cif_files):,} files into {flat_dir}/")

    for f in cif_files:
        f.rename(flat_dir / f.name)

    # Remove now-empty two-letter subdirectories
    for subdir in sorted(MIRROR_DIR.iterdir()):
        if subdir.is_dir() and subdir != flat_dir:
            try:
                subdir.rmdir()
            except OSError:
                pass  # not empty — skip

    print("  Done.")

### Main ###

def download_mmcif():
    ids = fetch_pdb_ids()

    id_file = Path(f'{DATA_DIR}/metadata/xray_lt3A_pdb_ids.txt')
    id_file.write_text("\n".join(sorted(ids)))
    print(f"  PDB IDs saved to {id_file}")

    rsync_mirror()
    filter_mirror(ids)
    decompress_all()
    flatten_mirror()

    print(f"\nAll structures in: {MIRROR_DIR / 'all'}/")
