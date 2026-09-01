"""Unified minimum-spanning-tree layer for CROWN cluster metrics.

Every CROWN similarity metric is clustered by *single linkage*, and single-linkage
clustering is exactly the minimum spanning tree (MST) of the distance graph d = 1 - sim:
cutting the dendrogram at height h and taking connected components of {edges with d <= h}
is the same operation, and the MST preserves every such cut. So one MST per metric is all
you ever need to relabel at any threshold, with no n x n matrix in sight.

The individual clustering scripts already emit an MST each, but in three different shapes:

    cluster_mmseqs.py -> crown_mst.parquet       (metric col: seq-sim, pocket-sim)  nodes = basename
    cluster_plec.py   -> CROWN_plec_mst.parquet  (pli-sim)                           nodes = basename
    cluster_ecfp4.py  -> CROWN_ecfp4_mst.parquet (lig-sim)                           nodes = lig_name

This module gives them a single canonical form and one place for the clustering logic:

  * SOURCES        - the registry: for each metric, where its MST lives, which id list
                     recovers its singletons, which metadata column its nodes map onto,
                     and the edge_floor below which a cut is not valid.
  * combine_msts() - stack every available source into one CROWN_combined_mst.parquet
                     (columns: metric, id1, id2, similarity) plus a sidecar JSON that
                     carries each metric's node list and floor. Self-contained: those two
                     files are all the reviewer (or make_clusters.py) needs.
  * single_linkage_labels() - the one clustering implementation, shared by everything.

Verified against scipy.cluster.hierarchy single linkage: the MST-cut partition is
identical to linkage(method='single') + fcluster(criterion='distance') for every metric.
"""

import os
import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.config import DATA_DIR

META_DIR = f'{DATA_DIR}/metadata'
COMBINED_MST = f'{META_DIR}/CROWN_combined_mst.parquet'
COMBINED_META = f'{META_DIR}/CROWN_combined_mst_meta.json'

# --------------------------------------------------------------------- registry
#
# One row per metric. `mst_file` may be shared (seq-sim and pocket-sim both live in
# crown_mst.parquet, distinguished by a `metric` column -> from_metric_col=True).
#
# edge_floor is the LOWEST threshold at which a cut is exact. The sequence pipeline only
# stores complex pairs whose best metric >= edge_floor (0.3 by default), so its MST is a
# valid single-linkage tree for cuts >= 0.3 but NOT below -- pairs beneath the floor were
# never scored. PLEC and ECFP4 build the MST over the complete graph, so any cut in [0, 1]
# is exact and their floor is 0.0. If you re-run cluster_mmseqs with a different edge_floor,
# set it here (it is not persisted by that script).
SOURCES = {
    'seq-sim': dict(
        mst_file=f'{META_DIR}/crown_mst.parquet',
        ids_file=f'{META_DIR}/crown_complex_ids.json',
        id_column='basename', edge_floor=0.3, from_metric_col=True),
    'pocket-sim': dict(
        mst_file=f'{META_DIR}/crown_mst.parquet',
        ids_file=f'{META_DIR}/crown_complex_ids.json',
        id_column='basename', edge_floor=0.3, from_metric_col=True),
    'pli-sim': dict(
        mst_file=f'{META_DIR}/CROWN_plec_mst.parquet',
        ids_file=f'{META_DIR}/CROWN_plec_ids.json',
        id_column='basename', edge_floor=0.0, from_metric_col=False),
    'lig-sim': dict(
        mst_file=f'{META_DIR}/CROWN_ecfp4_mst.parquet',
        ids_file=f'{META_DIR}/CROWN_ecfp4_ids.json',
        id_column='lig_name', edge_floor=0.0, from_metric_col=False),
}


# ------------------------------------------------------------------ clustering

def single_linkage_labels(mst_edges, ids, threshold):
    """Flat single-linkage labels at `threshold`, cut from a merge tree/forest.

    Keep only merges with similarity >= threshold and take connected components; nodes
    with no surviving edge fall out as singletons because `ids` supplies the full node
    list. Returns an int label array aligned to `ids` (labels[k] is the cluster of ids[k]).

    Valid only for threshold >= the metric's edge_floor (see SOURCES). Cheap enough to call
    repeatedly in a sweep -- it touches only above-threshold edges.

    Parameters
    ----------
    mst_edges : DataFrame with columns id1, id2, similarity (already filtered to ONE metric)
    ids       : list of node ids in this metric's namespace (recovers singletons)
    threshold : float similarity cut-off
    """
    idx = {c: i for i, c in enumerate(ids)}
    n = len(ids)
    sel = mst_edges[mst_edges['similarity'] >= threshold]

    i = sel['id1'].map(idx).to_numpy()
    j = sel['id2'].map(idx).to_numpy()
    # A NaN here means an edge endpoint is not in `ids`: almost always the wrong id list
    # for this metric (e.g. complex basenames against the ligand namespace). Fail loudly
    # rather than silently dropping edges and under-clustering.
    if len(i) and (pd.isna(i).any() or pd.isna(j).any()):
        missing = set(sel['id1']).union(sel['id2']) - set(idx)
        raise KeyError(
            f"{len(missing)} MST node(s) absent from the id list for this metric, "
            f"e.g. {list(missing)[:5]}. Are you using the right ids/namespace?")
    i = i.astype(np.int32)
    j = j.astype(np.int32)

    data = np.ones(len(i), dtype=np.int8)
    graph = coo_matrix((data, (i, j)), shape=(n, n))
    _, labels = connected_components(graph, directed=False, connection='weak')
    return labels


# ------------------------------------------------------------------ combine

def available_metrics():
    """Metrics whose source MST file is present on disk, in registry order."""
    return [m for m, s in SOURCES.items() if os.path.exists(s['mst_file'])]


def _read_source_mst(metric):
    """Load one metric's MST edges as a [id1, id2, similarity] DataFrame from its source."""
    s = SOURCES[metric]
    df = pd.read_parquet(s['mst_file'])
    if s['from_metric_col']:
        df = df[df['metric'] == metric]
    return df[['id1', 'id2', 'similarity']].reset_index(drop=True)


def _read_ids(metric):
    with open(SOURCES[metric]['ids_file']) as fh:
        return json.load(fh)


def combine_msts(metrics=None, floors=None, verbose=True):
    """Stack every available metric's MST into one canonical pair of files.

    Writes:
      CROWN_combined_mst.parquet       columns [metric, id1, id2, similarity]
      CROWN_combined_mst_meta.json     per-metric {id_column, edge_floor, n_nodes,
                                       n_edges, sim_min, sim_max, ids:[...]}

    The node lists are stored inline in the JSON so the two files are self-contained:
    they recover singletons without needing the original per-metric id files. Missing
    sources are skipped (so you can combine before ECFP4 has run, then re-combine).

    Parameters
    ----------
    metrics : optional subset of SOURCES keys (default: all present on disk)
    floors  : optional {metric: edge_floor} overriding the registry defaults
    """
    metrics = metrics or available_metrics()
    floors = floors or {}
    if not metrics:
        raise FileNotFoundError(
            "No source MST files found. Run the clustering scripts first "
            "(cluster_mmseqs / cluster_plec / cluster_ecfp4).")

    parts, meta = [], {}
    for m in metrics:
        edges = _read_source_mst(m)
        ids = _read_ids(m)
        part = edges.copy()
        part.insert(0, 'metric', m)
        parts.append(part)
        sim = edges['similarity']
        meta[m] = {
            'id_column': SOURCES[m]['id_column'],
            'edge_floor': float(floors.get(m, SOURCES[m]['edge_floor'])),
            'n_nodes': len(ids),
            'n_edges': int(len(edges)),
            'sim_min': float(sim.min()) if len(sim) else None,
            'sim_max': float(sim.max()) if len(sim) else None,
            'ids': ids,
        }
        if verbose:
            print(f"  {m:12s} {len(edges):>8,} edges over {len(ids):>8,} nodes "
                  f"(floor {meta[m]['edge_floor']:.2f}, "
                  f"sim {meta[m]['sim_min']:.3f}-{meta[m]['sim_max']:.3f})")

    combined = pd.concat(parts, ignore_index=True)
    os.makedirs(META_DIR, exist_ok=True)
    combined.to_parquet(COMBINED_MST, index=False)
    with open(COMBINED_META, 'w') as fh:
        json.dump(meta, fh)
    if verbose:
        print(f"Wrote {COMBINED_MST} ({len(combined):,} edges, {len(metrics)} metrics)")
        print(f"Wrote {COMBINED_META}")
    return combined, meta


# ------------------------------------------------------------------ load

def load_combined(build_if_missing=True):
    """Return (combined_mst_df, meta_dict), building the combined files if absent."""
    if not (os.path.exists(COMBINED_MST) and os.path.exists(COMBINED_META)):
        if not build_if_missing:
            raise FileNotFoundError(
                f"{COMBINED_MST} not found. Run combine_msts() first.")
        return combine_msts()
    mst = pd.read_parquet(COMBINED_MST)
    with open(COMBINED_META) as fh:
        meta = json.load(fh)
    return mst, meta


def labels_for(metric, threshold, mst=None, meta=None, force=False):
    """Convenience: single-linkage labels for one metric/threshold off the combined MST.

    Returns (ids, labels, id_column). Raises if threshold < edge_floor unless force=True.
    """
    if mst is None or meta is None:
        mst, meta = load_combined()
    if metric not in meta:
        raise KeyError(f"metric '{metric}' not available; have {list(meta)}")
    info = meta[metric]
    floor = info['edge_floor']
    if threshold < floor and not force:
        raise ValueError(
            f"threshold {threshold} is below {metric}'s edge_floor {floor}: the MST does "
            f"not contain the edges needed for a cut that low, so labels would be wrong. "
            f"Use a threshold >= {floor}, or pass force=True if you know the floor is 0.")
    edges = mst[mst['metric'] == metric]
    labels = single_linkage_labels(edges, info['ids'], threshold)
    return info['ids'], labels, info['id_column']
