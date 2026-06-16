import numpy as np
import pandas as pd
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
    return valid_labels, fps, failed

def save_fingerprints(valid_labels, fps, output_path):
    """Save fingerprints and labels to an npz file."""
    np.savez_compressed(
        output_path,
        labels=np.array(valid_labels),
        fingerprints=fps,
    )
    print(f"Saved fingerprints to {output_path} ({fps.shape[0]} x {fps.shape[1]})")

if __name__ == '__main__':
    df = pd.read_parquet(f'{DATA_DIR}/metadata/CROWN_full_annotated.parquet')
    basename_list = df['basename'].tolist()

    fp_path = f'{DATA_DIR}/metadata/CROWN_plec_fingerprints.npz'

    # Step 1: Compute and save fingerprints
    valid_labels, fps, failed = compute_fingerprints(basename_list, n_jobs=32, size = 32768)
    save_fingerprints(valid_labels, fps, fp_path)
