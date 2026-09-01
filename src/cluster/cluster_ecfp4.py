"""ECFP4 single-linkage clustering for CROWN ligands, as an MST.

Nodes here are UNIQUE LIGANDS (deduplicated by lig_name), not complexes -- the ligand
namespace, mapped back onto metadata via the `lig_name` column. Single-linkage clustering
is the minimum spanning tree of the distance graph d = 1 - Tanimoto, so instead of storing
a dense n x n similarity matrix (the old CROWN_ligsim.h5, which is the thing that OOMs on
large ligand sets) we store only the MST: at most n-1 edges. Re-cluster at any threshold in
[0, 1] straight from the tree, with no matrix rebuilt.

The MST is built by Prim's algorithm with distances computed one row at a time via RDKit's
C-level BulkTanimotoSimilarity, holding only three length-n arrays. Same shape and output
convention as cluster_plec.py, so make_clusters.py / crown_mst.py treat all metrics alike.

Outputs (all tiny):
  CROWN_ecfp4_mst.parquet      - merge tree: id1, id2, similarity  (nodes = lig_name)
  CROWN_ecfp4_ids.json         - node order (recovers singletons)
  CROWN_ecfp4_clusters.parquet - lig_name + cluster labels at 0.5 / 0.7 / 0.9
"""

import json

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from src.config import DATA_DIR
from src.cluster.crown_mst import single_linkage_labels

META_DIR = f'{DATA_DIR}/metadata'


# --------------------------------------------------------------------- fingerprints

def compute_ecfp4_fingerprints(smiles_list, labels, radius=2, n_bits=2048):
    """ECFP4 (Morgan radius=2) bit-vector fingerprints. Returns (fps, valid_labels):
    fps is a list of RDKit ExplicitBitVect, valid_labels the lig_names that parsed."""
    fps, valid_labels = [], []
    for smi, lab in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Warning: invalid SMILES for {lab}: {smi}")
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))
        valid_labels.append(lab)
    print(f"Fingerprinted {len(fps)}/{len(smiles_list)} ligands")
    return fps, valid_labels


# ------------------------------------------------------------------- streaming MST

def build_ecfp4_mst(fps, labels):
    """Exact single-linkage MST over Tanimoto distance via Prim's algorithm.

    Each iteration adds one node and refreshes every outside node's distance to the tree
    using a single BulkTanimotoSimilarity call (RDKit computes the whole row in C). Only
    length-n arrays are held -- never the n x n matrix. Compute is n^2/2 Tanimoto
    evaluations, the same as a full matrix, but peak memory is O(n).

    Returns a DataFrame [id1, id2, similarity] with n-1 rows, where `similarity` is the
    Tanimoto merge height at which each node joins the tree. The graph is complete (any two
    fingerprints have a defined Tanimoto, 0 if disjoint), so the tree spans every node and
    the cut is exact for any threshold in [0, 1].
    """
    n = len(fps)
    if n < 2:
        return pd.DataFrame({'id1': [], 'id2': [], 'similarity': []})

    in_tree = np.zeros(n, dtype=bool)
    min_d = np.full(n, np.inf, dtype=np.float64)   # min distance from each node to the tree
    nearest = np.zeros(n, dtype=np.int64)          # tree node achieving that min
    e_src = np.empty(n - 1, dtype=np.int64)
    e_dst = np.empty(n - 1, dtype=np.int64)
    e_sim = np.empty(n - 1, dtype=np.float64)

    cur = 0
    in_tree[0] = True
    for e in range(n - 1):
        # Whole row of Tanimoto similarities from `cur` to every node, computed in C.
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[cur], fps),
                          dtype=np.float64)
        d = 1.0 - sims
        upd = (~in_tree) & (d < min_d)
        min_d[upd] = d[upd]
        nearest[upd] = cur

        # Closest outside node = the next single-linkage merge.
        masked = np.where(in_tree, np.inf, min_d)
        best = int(np.argmin(masked))
        e_src[e] = nearest[best]
        e_dst[e] = best
        e_sim[e] = 1.0 - masked[best]
        in_tree[best] = True
        cur = best

        if (e + 1) % 2000 == 0:
            print(f"  MST: {e + 1}/{n - 1} edges")

    return pd.DataFrame({
        'id1': [labels[s] for s in e_src],
        'id2': [labels[d] for d in e_dst],
        'similarity': e_sim.astype(np.float32),
    })


# ---------------------------------------------------------------------- pipeline

def cluster_ecfp4(thresholds=(0.5, 0.7, 0.9)):
    df = pd.read_parquet(f'{META_DIR}/CROWN_metadata.parquet')

    # One fingerprint per unique ligand identifier.
    ligands = df.drop_duplicates(subset='lig_name')
    fps, labels = compute_ecfp4_fingerprints(
        ligands['SMILES'].tolist(), ligands['lig_name'].tolist())

    print(f"Building MST over {len(labels)} unique ligands...")
    mst = build_ecfp4_mst(fps, labels)
    print(f"Done. MST has {len(mst)} edges.")

    mst.to_parquet(f'{META_DIR}/CROWN_ecfp4_mst.parquet', index=False)
    with open(f'{META_DIR}/CROWN_ecfp4_ids.json', 'w') as fh:
        json.dump(labels, fh)

    # Convenience label columns (identical to scipy single-linkage), mirroring PLEC output.
    cols = {'lig_name': labels}
    for t in thresholds:
        cols[f'{t} lig-sim cluster'] = single_linkage_labels(mst, labels, t)
    pd.DataFrame(cols).to_parquet(f'{META_DIR}/CROWN_ecfp4_clusters.parquet', index=False)
    print(f"Wrote CROWN_ecfp4_mst.parquet, CROWN_ecfp4_ids.json, "
          f"CROWN_ecfp4_clusters.parquet")


# ------------------------------------------------------- optional: dense matrix (legacy)

def build_ligand_similarity_matrix(smiles_list, labels):
    """LEGACY: full dense n x n Tanimoto DataFrame. Kept for anyone who still wants the
    matrix (e.g. heatmaps); the pipeline no longer needs it and it does not scale. Prefer
    build_ecfp4_mst for clustering."""
    fps, valid_labels = compute_ecfp4_fingerprints(smiles_list, labels)
    n = len(fps)
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if i % 100 == 0:
            print(f"Running {i}/{n}...")
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        sim_matrix[i, i + 1:] = sims
        sim_matrix[i + 1:, i] = sims
        sim_matrix[i, i] = 1.0
    return pd.DataFrame(sim_matrix, index=valid_labels, columns=valid_labels)
