import numpy as np
import numba as nb
from numba import prange
import h5py

from src.config import DATA_DIR

def load_fingerprints(input_path):
    """Load fingerprints and labels from an npz file."""
    data = np.load(input_path, allow_pickle=True)
    labels = data['labels'].tolist()
    fps = data['fingerprints']
    print(f"Loaded fingerprints from {input_path} ({fps.shape[0]} x {fps.shape[1]})")
    return labels, fps

@nb.njit(nb.float32[:, :](nb.uint8[:, :], nb.int32[:]), parallel=True, cache=True)
def tanimoto_full(packed, counts):
    """
    Full pairwise Tanimoto from packed uint8 fingerprints.
    Only computes upper triangle, mirrors to lower.
    """
    n = packed.shape[0]
    b = packed.shape[1]
    sim = np.zeros((n, n), dtype=np.float32)

    for i in prange(n):
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
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('similarity', data=sim.astype(np.float16),
                         compression='gzip', compression_opts=1)
        f.create_dataset('labels', data=np.array(valid_labels, dtype='S'))
    print(f"Saved to {output_path}")

    return sim


if __name__ == '__main__':
    labels, fps = load_fingerprints(f'{DATA_DIR}/metadata/CROWN_plec_fingerprints.npz')
    sim = build_similarity_matrix(
        labels, fps,
        output_path=f'{DATA_DIR}/metadata/CROWN_plisim.h5',
    )
