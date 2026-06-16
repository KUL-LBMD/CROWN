import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import h5py
import json

from src.config import DATA_DIR

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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

if __name__ == '__main__':

    sim_path = f'{DATA_DIR}/metadata/CROWN_plisim.h5'

    with h5py.File(sim_path, 'r') as f:
        labels = [l.decode() for l in f['labels'][:]]
        sim = f['similarity'][:].astype(np.float16)

    sim_df = pd.DataFrame(sim, index=labels, columns=labels)
    del labels, sim

    plec_cluster_dict = build_cluster_dict(sim_df, 0.50)

    with open(f'{DATA_DIR}/metadata/plec_cluster_dict_50.json', 'w') as f:
        json.dump(plec_cluster_dict, f, cls=NumpyEncoder)

    plec_cluster_dict = build_cluster_dict(sim_df, 0.70)

    with open(f'{DATA_DIR}/metadata/plec_cluster_dict_70.json', 'w') as f:
        json.dump(plec_cluster_dict, f, cls=NumpyEncoder)

    plec_cluster_dict = build_cluster_dict(sim_df, 0.90)

    with open(f'{DATA_DIR}/metadata/plec_cluster_dict_90.json', 'w') as f:
        json.dump(plec_cluster_dict, f, cls=NumpyEncoder)
