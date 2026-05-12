from src.config import DATA_DIR

import os
import pandas as pd
import numpy as np
from biopandas.mol2 import PandasMol2
from scipy.spatial import KDTree

def calculate_rmsd(subdir):

    ligand_df = PandasMol2.read_mol2(f'{DATA_DIR}/mol2_files/{subdir}/ligand.mol2').df
    ligand_min_df = PandasMol2.read_mol2(f'{DATA_DIR}/mol2_files/{subdir}/ligand_minimized.mol2').df
    receptor_df = PandasMol2.read_mol2(f'{DATA_DIR}/mol2_files/{subdir}/receptor.mol2').df
    receptor_min_df = PandasMol2.read_mol2(f'{DATA_DIR}/mol2_files/{subdir}/receptor_minimized.mol2').df

    lig_df = ligand_df[~ligand_df['atom_type'].str.startswith('H')]
    lig_min_df = ligand_min_df[~ligand_min_df['atom_type'].str.startswith('H')]
    prot_df = receptor_df[~receptor_df['atom_type'].str.startswith('H')]
    prot_min_df = receptor_min_df[~receptor_min_df['atom_type'].str.startswith('H')]
    h_df = receptor_df[receptor_df['atom_type'].str.startswith('H')]
    h_min_df = receptor_min_df[receptor_min_df['atom_type'].str.startswith('H')]

    lig_coords = lig_df[['x', 'y', 'z']].values
    lig_min_coords = lig_min_df[['x', 'y', 'z']].values
    h_coords = h_df[['x', 'y', 'z']].values
    h_min_coords = h_min_df[['x', 'y', 'z']].values

    prot_coords = prot_df[['x', 'y', 'z']].values
    prot_min_coords = prot_min_df[['x', 'y', 'z']].values

    # Divide prot coords into pocket and scaffold
    lig_tree = KDTree(lig_coords)
    idx = lig_tree.query_ball_point(prot_coords, r = 6)
    mask = np.array([len(i) > 0 for i in idx]) # Check which B coords have close neighbors in A

    pocket_coords = prot_coords[mask]
    pocket_min_coords = prot_min_coords[mask]
    scaffold_coords = prot_coords[~mask]
    scaffold_min_coords = prot_min_coords[~mask]

    lig_rmsd = np.sqrt(np.mean(((lig_coords - lig_min_coords)**2).sum(-1)))
    h_rmsd = np.sqrt(np.mean(((h_coords - h_min_coords)**2).sum(-1)))
    pocket_rmsd = np.sqrt(np.mean(((pocket_coords - pocket_min_coords)**2).sum(-1)))
    scaffold_rmsd = np.sqrt(np.mean(((scaffold_coords - scaffold_min_coords)**2).sum(-1)))

    return {'basename': subdir, 'Ligand_RMSD': lig_rmsd, 'Pocket_RMSD': pocket_rmsd, 'Scaffold_RMSD': scaffold_rmsd, 'H_RMSD': h_rmsd}

if __name__ == '__main__':
    list_of_dicts = []
    for subdir in os.listdir(f'{DATA_DIR}/mol2_files'):
        list_of_dicts.append(calculate_rmsd(subdir))

    df = pd.DataFrame(list_of_dicts)
    df.to_csv(f'{DATA_DIR}/metadata/CROWN_rmsd.csv')

