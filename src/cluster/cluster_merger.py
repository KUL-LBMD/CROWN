import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.config import DATA_DIR

def build_cluster_dict(sim_df, sim_level):
    """
    Parameters
    ----------
    sim_df : pd.DataFrame
        N x N symmetric similarity matrix, index names = column names.
    sim_level : float
        Cut-off to be in the same similarity cluster.

    Returns
    -------
    cluster_dict : dict
        Maps each entry to a cluster label.
    """
    # Convert similarity to distance (assuming similarity in [0, 1])
    dist_matrix = 1 - sim_df.values
    np.fill_diagonal(dist_matrix, 0)

    # squareform expects the upper triangle as a condensed vector
    condensed_dist = squareform(dist_matrix, checks=False)

    # Single-linkage clustering on the distance matrix
    Z = linkage(condensed_dist, method='single')

    # fcluster threshold is a distance: items with similarity > sim_level
    # correspond to distance < (1 - sim_level)
    labels = fcluster(Z, t=1 - sim_level, criterion='distance')

    index_list = sim_df.index.tolist()
    cluster_dict = dict(zip(index_list, labels))

    return cluster_dict

def map_clusters(df, cluster_dict, in_col_name, out_col_name):
    # Start labels for unmapped IDs above the existing max
    next_label = max(cluster_dict.values()) + 1 if cluster_dict else 0

    def assign_cluster(uid):
        nonlocal next_label
        if uid in cluster_dict:
            return cluster_dict[uid]
        else:
            label = next_label
            next_label += 1
            return label

    df = df.copy()
    df[out_col_name] = df[in_col_name].map(assign_cluster)
    return df

def merge_clusters():

    metadata_df = pd.read_parquet(f'{DATA_DIR}/metadata/CROWN_metadata.parquet')

    # 2. Ligand similarity
    sim_df = pd.read_hdf(f'{DATA_DIR}/metadata/CROWN_ligsim.h5')
    cluster_dict_50 = build_cluster_dict(sim_df, 0.50)
    cluster_dict_70 = build_cluster_dict(sim_df, 0.70)
    cluster_dict_90 = build_cluster_dict(sim_df, 0.90)
    metadata_df = map_clusters(metadata_df, cluster_dict_50, 'lig_name', '0.5 ligsim cluster')
    metadata_df = map_clusters(metadata_df, cluster_dict_70, 'lig_name', '0.7 ligsim cluster')
    metadata_df = map_clusters(metadata_df, cluster_dict_90, 'lig_name', '0.9 ligsim cluster')

    # 3. PLI similarity
    sim_df = pd.read_hdf(f'{DATA_DIR}/metadata/CROWN_plisim.h5')
    cluster_dict_50 = build_cluster_dict(sim_df, 0.50)
    cluster_dict_70 = build_cluster_dict(sim_df, 0.70)
    cluster_dict_90 = build_cluster_dict(sim_df, 0.90)
    metadata_df = map_clusters(metadata_df, cluster_dict_50, 'basename', '0.5 plisim cluster')
    metadata_df = map_clusters(metadata_df, cluster_dict_70, 'basename', '0.7 plisim cluster')
    metadata_df = map_clusters(metadata_df, cluster_dict_90, 'basename', '0.9 plisim cluster')

    # 1. Sequence similarity
    seq_df = pd.read_csv(f'{DATA_DIR}/metadata/crown_seq_cluster_labels.csv')
    metadata_df = metadata_df.merge(seq_df, on = ['basename'])

    metadata_df.to_parquet(f'{DATA_DIR}/metadata/CROWN_metadata.parquet', index = False)
