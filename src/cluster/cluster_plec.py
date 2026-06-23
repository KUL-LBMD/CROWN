import numpy as np
import numba as nb
import pandas as pd
import os
import oddt
from oddt.fingerprints import PLEC
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed

from src.config import DATA_DIR

def get_plec_fingerprint(basename, size = 32768):
    """Compute PLEC fingerprint for a single complex. Returns (basename, fp) or (basename, None)."""
    try:
        prot_path = f'{DATA_DIR}/complexes/{basename}/receptor_minimized.pdb'
        lig_path = f'{DATA_DIR}/complexes/{basename}/ligand_minimized.sdf'

        prot = next(oddt.toolkit.readfile('pdb', prot_path))
        lig = next(oddt.toolkit.readfile('sdf', lig_path))
        prot.protein = True

        fp = PLEC(lig, prot, size = 32768, sparse = False, count_bits = False).flatten()

        return basename, fp
    except Exception as e:
        print(f"Warning: failed on {basename}: {e}")
        return basename, None

def compute_fingerprints(basename_list, n_jobs=-1, size=4096):
    """Compute PLEC fingerprints in parallel using joblib."""
    print(f"Computing PLEC fingerprints for {len(basename_list)} complexes using {n_jobs} jobs...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(get_plec_fingerprint)(basename, size) for basename in basename_list)

    valid_labels = []
    fps = []
    failed = []
    wrong_size = []
    for basename, fp in results:
        if fp is not None:
            if fp.shape[0] == size:
                valid_labels.append(basename)
                fps.append(fp)
            else:
                wrong_size.append((basename, fp.shape[0]))
        else:
            failed.append(basename)

    print(f"Successfully computed {len(fps)}/{len(basename_list)} fingerprints")
    if failed:
        print(f"Failed: {len(failed)} complexes")
    if wrong_size:
        print(f"Wrong size (excluded): {len(wrong_size)} complexes")
        for name, s in wrong_size[:5]:
            print(f"  {name}: got {s}, expected {size}")

    fps = np.vstack(fps)  # safer than np.array for this case
    return valid_labels, fps

@nb.njit(nb.float32[:, :](nb.uint8[:, :], nb.int32[:]), parallel=True, cache=True)
def tanimoto_full(packed, counts):
    """
    Full pairwise Tanimoto from packed uint8 fingerprints.
    Only computes upper triangle, mirrors to lower.
    """
    n = packed.shape[0]
    b = packed.shape[1]
    sim = np.zeros((n, n), dtype=np.float32)

    for i in nb.prange(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            inter = nb.int32(0)
            for k in range(b):
                x = nb.uint8(packed[i, k] & packed[j, k])
                while x:
                    inter += 1
                    x &= nb.uint8(x - nb.uint8(1))
            union = counts[i] + counts[j] - inter
            val = nb.float32(inter) / nb.float32(union) if union > 0 else nb.float32(0.0)
            sim[i, j] = val
            sim[j, i] = val
    return sim

def build_similarity_matrix(valid_labels, fps, output_path):
    """Compute full 150k x 150k Tanimoto in memory, save to HDF5."""
    fps_packed = np.packbits(fps.astype(np.uint8), axis=1)
    n = len(valid_labels)
    print(f"Packed: {fps_packed.shape} — computing {n}x{n} Tanimoto...")

    # Precompute per-fingerprint bit counts
    lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
    counts = lut[fps_packed].sum(axis=1).astype(np.int32)
    fps_packed = fps_packed.astype(np.uint8)

    # Warm up Numba JIT
    _ = tanimoto_full(fps_packed[:2], counts[:2])
    print("JIT compiled. Running full computation...")

    sim = tanimoto_full(fps_packed, counts)
    print(f"Done. Matrix shape: {sim.shape}")

    # Save
    sim_df = pd.DataFrame(sim.astype(np.float16), index = valid_labels, columns = valid_labels)
    sim_df.to_hdf(output_path, key = 'sim', complevel = 5, complib = 'blosc')
    print(f"Saved to {output_path}")

def cluster_plec(num_cores = 1):
	basename_list = os.listdir(f'{DATA_DIR}/complexes')
	labels, fps = compute_fingerprints(basename_list, n_jobs = num_cores, size = 32768)
	build_similarity_matrix(labels, fps, output_path=f'{DATA_DIR}/metadata/CROWN_plisim.h5')
