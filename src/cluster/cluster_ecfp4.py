import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from src.config import DATA_DIR

def build_ligand_similarity_matrix(smiles_list, labels):
    # Generate ECFP4 fingerprints (radius=2, 2048 bits)
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
            valid_idx.append(i)
        else:
            print(f"Warning: invalid SMILES at index {i}: {smi}")

    n = len(fps)
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    # Compute pairwise Tanimoto (symmetric, so only upper triangle)
    for i in range(n):

        if i % 100 == 0:
                print(f"Running {i}/{n}...")

        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        sim_matrix[i, i+1:] = sims
        sim_matrix[i+1:, i] = sims
        sim_matrix[i, i] = 1.0

    valid_labels = [labels[i] for i in valid_idx]
    return pd.DataFrame(sim_matrix, index=valid_labels, columns=valid_labels)

def cluster_ecfp4():
    df = pd.read_parquet(f'{DATA_DIR}/metadata/CROWN_metadata.parquet')

    # Deduplicate by ligand identifier
    ligands = df.drop_duplicates(subset='lig_name')
    smiles_list = ligands['SMILES'].tolist()
    labels = ligands['lig_name'].tolist()

    sim_df = build_ligand_similarity_matrix(smiles_list, labels)
    sim_df.to_hdf(f'{DATA_DIR}/metadata/CROWN_ligsim.h5', key='sim', complevel=5, complib='blosc')
    print(f"Saved {sim_df.shape[0]}x{sim_df.shape[1]} similarity matrix")
