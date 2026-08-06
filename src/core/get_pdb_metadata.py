import csv
import time
from pathlib import Path
import requests
import pandas as pd

from src.config import DATA_DIR

GRAPHQL_URL = "https://data.rcsb.org/graphql"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
METADATA_CSV = Path(f'{DATA_DIR}/metadata/pdb_xray_metadata.csv')
CRYO_CSV = Path(f'{DATA_DIR}/metadata/pdb_cryo_metadata.csv')
BATCH_SIZE = 1000  # GraphQL batch size

def fetch_xray_pdb_ids() -> set[str]:
    """Query RCSB for X-ray structures < 3 Å and write pdb_id, resolution, R-free to CSV.

    Still returns the set of PDB IDs so the rsync-filtering step downstream keeps working.
    """
    print("Querying RCSB Search API...")
    search_query = {
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
    resp = requests.post(SEARCH_URL, json=search_query)
    resp.raise_for_status()
    ids = [hit["identifier"] for hit in resp.json()["result_set"]]
    print(f"  Found {len(ids):,} matching entries")

    # Fetch resolution and R-free from the Data API
    gql = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        rcsb_entry_info { resolution_combined }
        refine { ls_R_factor_R_free }
      }
    }
    """
    print(f"  Fetching metadata in batches of {BATCH_SIZE}...")
    with METADATA_CSV.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pdb_id", "resolution", "r_free"])
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i : i + BATCH_SIZE]
            r = requests.post(
                GRAPHQL_URL,
                json={"query": gql, "variables": {"ids": batch}},
            )
            r.raise_for_status()
            for entry in r.json()["data"]["entries"] or []:
                if entry is None:
                    continue  # obsolete / withdrawn
                res_list = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
                resolution = res_list[0] if res_list else ""
                refine = entry.get("refine") or []
                rfree = refine[0].get("ls_R_factor_R_free") if refine else ""
                writer.writerow([entry["rcsb_id"].lower(), resolution, rfree if rfree is not None else ""])
            print(f"    {min(i + BATCH_SIZE, len(ids)):,} / {len(ids):,}")
            time.sleep(0.1)  # be polite

    print(f"  Wrote metadata to {METADATA_CSV}")

def fetch_cryo_pdb_ids() -> set[str]:
    """Query RCSB for X-ray structures < 3 Å and write pdb_id, resolution, R-free to CSV.

    Still returns the set of PDB IDs so the rsync-filtering step downstream keeps working.
    """
    print("Querying RCSB Search API...")
    search_query = {
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
    resp = requests.post(SEARCH_URL, json=search_query)
    resp.raise_for_status()
    ids = [hit["identifier"] for hit in resp.json()["result_set"]]
    print(f"  Found {len(ids):,} matching entries")

    # Fetch resolution and R-free from the Data API
    gql = """
    query($ids: [String!]!) {
        entries(entry_ids: $ids) {
        rcsb_id
        rcsb_entry_info { resolution_combined }
        }
    }
    """
    print(f"  Fetching metadata in batches of {BATCH_SIZE}...")
    with CRYO_CSV.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pdb_id", "resolution"])
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i : i + BATCH_SIZE]
            r = requests.post(
                GRAPHQL_URL,
                json={"query": gql, "variables": {"ids": batch}},
            )
            r.raise_for_status()
            for entry in r.json()["data"]["entries"] or []:
                if entry is None:
                    continue  # obsolete / withdrawn
                res_list = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
                resolution = res_list[0] if res_list else ""
                writer.writerow([entry["rcsb_id"].lower(), resolution if resolution is not None else ""])
            print(f"    {min(i + BATCH_SIZE, len(ids)):,} / {len(ids):,}")
            time.sleep(0.1)  # be polite

    print(f"  Wrote metadata to {CRYO_CSV}")

def get_pdb_metadata():

    fetch_xray_pdb_ids()
    fetch_cryo_pdb_ids()

    xray_df = pd.read_csv(METADATA_CSV)
    cryo_df = pd.read_csv(CRYO_CSV)

    xray_df['experimental_method'] = 'X-ray Crystallography'
    cryo_df['experimental_method'] = 'Cryo-EM'
    final_df = pd.concat([xray_df, cryo_df], axis = 0, ignore_index = True)
    final_df.to_csv(f'{DATA_DIR}/metadata/pdb_metadata.csv', index = False, float_format = '%.6f')
