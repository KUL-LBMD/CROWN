"""PLEC single-linkage clustering for CROWN, in O(n) memory.

The end goal is single-linkage cluster labels over all CROWN entries from PLEC
(protein-ligand) Tanimoto similarity. The obstacle is that the full n x n Tanimoto matrix
is enormous (150k^2 * 4 bytes ~ 90 GB as float32, plus a copy to write HDF5), so it gets
OOM-killed. Two ideas remove the memory ceiling without changing the science:

  1. Single-linkage clustering IS the minimum spanning tree of the distance graph
     (d = 1 - Tanimoto). Prim's algorithm builds that MST while computing distances one
     row at a time, holding only three length-n arrays -- never the n^2 matrix. Each pair
     distance is computed exactly once, so compute is the same O(n^2 * b) as a full matrix,
     but peak memory drops from O(n^2) to O(n). Because it runs over the complete graph,
     the MST is valid for cuts at ANY threshold in [0, 1].

  2. Fingerprints are bit-packed the moment they are computed, so we store 150k x (size/8)
     bytes (~600 MB at size=32768) instead of the dense 150k x size array.

Outputs (all tiny):
  CROWN_plec_mst.parquet      - the single-linkage merge tree: id1, id2, similarity
  CROWN_plec_ids.json         - node order (recovers singletons; here the graph is
                                connected so every node is in the tree)
  CROWN_plec_clusters.parquet - complex_id, cluster at the configured threshold

Cache:
  metadata/plec_fp_cache_{size}.npz - packed fingerprints keyed by width. Re-runs load
  this and only fingerprint basenames not already present, then go straight to the MST.

Re-cluster at any threshold straight from the MST (see `single_linkage_labels`) without
recomputing a single fingerprint or distance.
"""

import os
import json

import numpy as np
import numba as nb
import pandas as pd
import oddt
from oddt.fingerprints import PLEC
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed

from src.config import DATA_DIR

# uint8 -> popcount lookup, shared by the Numba kernel.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


# --------------------------------------------------------------------- fingerprints

def get_plec_fingerprint(basename, size):
    """Compute the PLEC fingerprint for one complex and bit-pack it immediately.

    Returns (basename, packed_uint8, n_bits). On failure packed is None and n_bits is -1;
    on an unexpected length packed is None and n_bits is the length actually produced.
    Packing here (size bits -> size/8 bytes) is what keeps the fingerprint matrix small.
    """
    try:
        prot_path = f'{DATA_DIR}/complexes/{basename}/receptor_minimized.pdb'
        lig_path = f'{DATA_DIR}/complexes/{basename}/ligand_minimized.sdf'

        prot = next(oddt.toolkit.readfile('pdb', prot_path))
        lig = next(oddt.toolkit.readfile('sdf', lig_path))
        prot.protein = True

        # NOTE: size is now honoured (the original hard-coded 32768 here, ignoring the arg).
        fp = PLEC(lig, prot, size=size, sparse=False, count_bits=False).astype(np.uint8).ravel()
        if fp.shape[0] != size:
            return basename, None, int(fp.shape[0])
        return basename, np.packbits(fp), size
    except Exception as e:
        print(f"Warning: failed on {basename}: {e}")
        return basename, None, -1


def _fp_cache_path(size):
    return f'{DATA_DIR}/metadata/plec_fp_cache_{size}.npz'


def _load_fp_cache(size):
    """Return {basename: packed_row} from the on-disk cache, or {} if absent/unreadable.
    The cache filename is keyed by `size`, so a width change never reuses stale rows."""
    path = _fp_cache_path(size)
    if not os.path.exists(path):
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        c_packed = data['packed']
        if c_packed.shape[1] != size // 8:
            print(f"[warn] cache {path} has wrong width; ignoring")
            return {}
        c_labels = data['labels'].astype(str)
        return {lab: c_packed[i] for i, lab in enumerate(c_labels)}
    except Exception as e:
        print(f"[warn] could not read cache {path}: {e}; recomputing")
        return {}


def _save_fp_cache(size, fps):
    """Persist {basename: packed_row} to the size-keyed cache as a single .npz."""
    path = _fp_cache_path(size)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = list(fps.keys())
    packed = np.vstack([fps[b] for b in names])
    np.savez(path, packed=packed, labels=np.array(names), size=np.int64(size))
    print(f"Cached {len(names)} fingerprints to {path}")


def compute_fingerprints(basename_list, n_jobs=-1, size=32768,
                         cache=True, refresh=False):
    """Compute + pack PLEC fingerprints in parallel, reusing a disk cache so re-runs only
    fingerprint what's new. Returns (labels, packed, counts): packed is (n, size//8) uint8,
    counts is the per-fingerprint bit count (int32).

    Incremental by default: only basenames absent from the cache are computed, which makes
    adding complexes to CROWN cheap. cache=False disables the cache; refresh=True ignores
    and overwrites it (full recompute).
    """
    fps = {} if refresh or not cache else _load_fp_cache(size)
    if fps:
        print(f"Loaded {len(fps)} cached fingerprints (size={size})")

    to_compute = [b for b in basename_list if b not in fps]
    print(f"{len(basename_list) - len(to_compute)} cached, "
          f"{len(to_compute)} to compute (size={size})")

    if to_compute:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(get_plec_fingerprint)(b, size) for b in to_compute)

        failed, wrong_size, n_ok = [], [], 0
        for basename, packed, nbits in results:
            if packed is not None:
                fps[basename] = packed
                n_ok += 1
            elif nbits == -1:
                failed.append(basename)
            else:
                wrong_size.append((basename, nbits))

        print(f"Successfully computed {n_ok}/{len(to_compute)} new fingerprints")
        if failed:
            print(f"Failed: {len(failed)} complexes")
        if wrong_size:
            print(f"Wrong size (excluded): {len(wrong_size)} complexes")
            for name, s in wrong_size[:5]:
                print(f"  {name}: got {s}, expected {size}")

        if cache:
            _save_fp_cache(size, fps)  # persist the union so nothing recomputes next run

    # Assemble outputs in the requested order, keeping only successfully-fingerprinted ones.
    labels = [b for b in basename_list if b in fps]
    packed = np.vstack([fps[b] for b in labels])          # (n, size//8) uint8, ~600 MB
    counts = _POPCOUNT[packed].sum(axis=1).astype(np.int32)
    return labels, packed, counts


# ------------------------------------------------------------------- streaming MST

@nb.njit(parallel=True, cache=True)
def prim_mst_tanimoto(packed, counts, pop):
    """Exact single-linkage MST over Tanimoto distance (d = 1 - Tanimoto) via Prim's
    algorithm with on-the-fly distances. Holds only length-n arrays: no n x n matrix is
    ever formed. Returns (edge_src, edge_dst, edge_sim) of length n-1, where edge_sim is
    the merge-height Tanimoto similarity at which each node joins the tree.

    Compute is n^2/2 Tanimoto evaluations (each pair once), the same as a full matrix; the
    inner per-node update is parallel, the argmin pick is a cheap serial reduction.
    """
    n = packed.shape[0]
    b = packed.shape[1]

    in_tree = np.zeros(n, dtype=np.bool_)
    min_d = np.full(n, np.inf, dtype=np.float32)   # min distance from each node to the tree
    nearest = np.zeros(n, dtype=np.int64)          # tree node achieving that min
    e_src = np.empty(n - 1, dtype=np.int64)
    e_dst = np.empty(n - 1, dtype=np.int64)
    e_sim = np.empty(n - 1, dtype=np.float32)

    cur = 0
    in_tree[0] = True
    for e in range(n - 1):
        ci = counts[cur]
        # Update each outside node's distance to the tree using the just-added node `cur`.
        for k in nb.prange(n):
            if not in_tree[k]:
                inter = 0
                for kk in range(b):
                    inter += pop[packed[cur, kk] & packed[k, kk]]
                union = ci + counts[k] - inter
                sim = np.float32(inter) / np.float32(union) if union > 0 else np.float32(0.0)
                d = np.float32(1.0) - sim
                if d < min_d[k]:
                    min_d[k] = d
                    nearest[k] = cur
        # Pick the closest outside node (single-linkage merge) -- serial argmin.
        best = -1
        best_d = np.float32(np.inf)
        for k in range(n):
            if (not in_tree[k]) and min_d[k] < best_d:
                best_d = min_d[k]
                best = k
        in_tree[best] = True
        e_src[e] = nearest[best]
        e_dst[e] = best
        e_sim[e] = np.float32(1.0) - best_d
        cur = best

    return e_src, e_dst, e_sim


def build_mst(labels, packed, counts):
    """Run Prim's kernel (with a tiny warm-up compile) and return the merge tree as a
    DataFrame [id1, id2, similarity]."""
    # Warm up the JIT on a 2-row slice so the progress/timing below reflects real work.
    _ = prim_mst_tanimoto(packed[:2].copy(), counts[:2].copy(), _POPCOUNT)
    print(f"JIT compiled. Building MST over {len(labels)} complexes...")

    src, dst, sim = prim_mst_tanimoto(packed, counts, _POPCOUNT)
    print(f"Done. MST has {len(src)} edges.")
    return pd.DataFrame({
        'id1': [labels[s] for s in src],
        'id2': [labels[d] for d in dst],
        'similarity': sim,
    })


# --------------------------------------------------------------------- clustering

def single_linkage_labels(mst, complex_ids, threshold):
    """Flat single-linkage labels at `threshold`: connected components of the MST keeping
    only merges with Tanimoto similarity >= threshold. Valid for any threshold in [0, 1]
    (the MST spans the complete graph). Returns an int label array aligned to complex_ids.
    Cheap enough to call repeatedly in a threshold sweep."""
    idx = {c: i for i, c in enumerate(complex_ids)}
    n = len(complex_ids)
    sel = mst[mst['similarity'] >= threshold]
    i = sel['id1'].map(idx).to_numpy().astype(np.int32)
    j = sel['id2'].map(idx).to_numpy().astype(np.int32)
    data = np.ones(len(i), dtype=np.int8)
    graph = coo_matrix((data, (i, j)), shape=(n, n))
    _, labels = connected_components(graph, directed=False, connection='weak')
    return labels


# ---------------------------------------------------------------------- pipeline

def cluster_plec(num_cores=-1, size=32768, cluster_threshold=0.5,
                 cache=True, refresh_cache=False):
    basename_list = os.listdir(f'{DATA_DIR}/complexes')

    labels, packed, counts = compute_fingerprints(
        basename_list, n_jobs=num_cores, size=size,
        cache=cache, refresh=refresh_cache)

    mst = build_mst(labels, packed, counts)
    del packed, counts  # free ~600 MB before writing anything

    meta = f'{DATA_DIR}/metadata'
    os.makedirs(meta, exist_ok=True)
    mst.to_parquet(f'{meta}/CROWN_plec_mst.parquet', index=False)
    with open(f'{meta}/CROWN_plec_ids.json', 'w') as fh:
        json.dump(labels, fh)

    clusters_50 = single_linkage_labels(mst, labels, 0.5)
    clusters_70 = single_linkage_labels(mst, labels, 0.7)
    clusters_90 = single_linkage_labels(mst, labels, 0.9)
    pd.DataFrame({'basename': labels, 'plec_50': clusters_50, 'plec_70': clusters_70, 'plec_90': clusters_90}).to_parquet(
        f'{meta}/CROWN_plec_clusters.parquet', index=False)
