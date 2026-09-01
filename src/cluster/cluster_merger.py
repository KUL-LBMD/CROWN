"""Merge every metric's single-linkage cluster labels onto CROWN_metadata.parquet.

All four metrics now come from MSTs and write a small cluster-label table each, so this is
just a series of merges -- no dense similarity matrix is loaded anywhere:

  sequence / pocket : crown_seq_cluster_labels.csv     merged on basename
  protein-ligand    : CROWN_plec_clusters.parquet      merged on basename
  ligand (ECFP4)    : CROWN_ecfp4_clusters.parquet     merged on lig_name

The ligand labels are identical to the old dense-matrix + scipy single-linkage result
(single linkage IS the MST), so switching to the MST changes memory use, not the science.

Note: ligand label columns are now named '{t} lig-sim cluster' (was '{t} ligsim cluster').
To re-cut any metric at a different threshold, use make_clusters.py rather than editing here.
"""

import numpy as np
import pandas as pd

from src.config import DATA_DIR

META_DIR = f'{DATA_DIR}/metadata'


def _fill_unmapped_singletons(df, label_cols):
    """Give every still-unlabeled row (e.g. a ligand whose SMILES failed to parse, so it
    never entered the MST) its own fresh singleton label, above the existing max -- the
    same behaviour the old map_clusters() had for unmapped IDs."""
    for col in label_cols:
        missing = df[col].isna()
        if missing.any():
            start = int(np.nanmax(df[col].to_numpy())) + 1 if df[col].notna().any() else 0
            df.loc[missing, col] = np.arange(start, start + int(missing.sum()))
        df[col] = df[col].astype('int64')
    return df


def merge_clusters():
    metadata_df = pd.read_parquet(f'{META_DIR}/CROWN_metadata.parquet')

    # 1. Sequence + pocket similarity (nodes = basename)
    seq_df = pd.read_csv(f'{META_DIR}/crown_seq_cluster_labels.csv')
    metadata_df = metadata_df.merge(seq_df, on='basename', how='left')

    # 2. Protein-ligand interaction similarity (nodes = basename)
    plec_df = pd.read_parquet(f'{META_DIR}/CROWN_plec_clusters.parquet')
    metadata_df = metadata_df.merge(plec_df, on='basename', how='left')

    # 3. Ligand similarity (ECFP4; nodes = lig_name)
    ecfp4_df = pd.read_parquet(f'{META_DIR}/CROWN_ecfp4_clusters.parquet')
    lig_cols = [c for c in ecfp4_df.columns if c != 'lig_name']
    metadata_df = metadata_df.merge(ecfp4_df, on='lig_name', how='left')
    metadata_df = _fill_unmapped_singletons(metadata_df, lig_cols)

    metadata_df.to_parquet(f'{META_DIR}/CROWN_metadata.parquet', index=False)
    print(f"Merged cluster labels for {len(metadata_df):,} complexes")
