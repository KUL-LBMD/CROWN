from src.config import DATA_DIR

import os
import io
import gemmi
import pandas as pd
import numpy as np
from biopandas.mol2 import PandasMol2
from scipy.spatial import KDTree
from joblib import Parallel, delayed

def _fix_mol2(path):
    with open(path) as f:
        lines = f.readlines()

    in_atom = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('@<TRIPOS>'):
            in_atom = stripped == '@<TRIPOS>ATOM'
            continue
        if in_atom and stripped:
            parts = line.split()
            if len(parts) == 8:                       # atom_name missing
                # parts = [id, x, y, z, type, subst_id, subst_name, charge]
                parts.insert(1, parts[4])             # use atom_type as a stand-in name
                lines[i] = '  '.join(parts) + '\n'
    return io.StringIO(''.join(lines))

def read_mol2_safe(path):
    pmol = PandasMol2()
    pmol.read_mol2_from_list(
        mol2_lines=_fix_mol2(path).readlines(),
        mol2_code=path,
    )
    return pmol

def calculate_rmsd(subdir):

    try:

        ligand_df = read_mol2_safe(f'{DATA_DIR}/mol2_files/{subdir}/ligand.mol2').df
        ligand_min_df = read_mol2_safe(f'{DATA_DIR}/mol2_files/{subdir}/ligand_minimized.mol2').df
        receptor_df = read_mol2_safe(f'{DATA_DIR}/mol2_files/{subdir}/receptor.mol2').df
        receptor_min_df = read_mol2_safe(f'{DATA_DIR}/mol2_files/{subdir}/receptor_minimized.mol2').df

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
        scaffold_init_coords = prot_coords[~mask]
        scaffold_init_min_coords = prot_min_coords[~mask]

        # Further subdivide scaffold into original atoms and fixed atoms
        structure = gemmi.read_structure(f'{DATA_DIR}/pdb/raw/{subdir}.pdb')
        og_coords = np.array([
            [atom.pos.x, atom.pos.y, atom.pos.z]
            for model in structure
            for chain in model
            for residue in chain
            for atom in residue
        ])

        og_tree = KDTree(og_coords)
        idx = og_tree.query_ball_point(scaffold_init_coords, r = 0.1)
        mask = np.array([len(i) > 0 for i in idx]) # Check which B coords have close neighbors in A

        if mask.shape[0] > 0:
            scaffold_coords = scaffold_init_coords[mask]
            scaffold_min_coords = scaffold_init_min_coords[mask]
            rebuilt_coords = scaffold_init_coords[~mask]
            rebuilt_min_coords = scaffold_init_min_coords[~mask]
        else:
            scaffold_coords = scaffold_init_coords
            scaffold_min_coords = scaffold_init_min_coords
            rebuilt_coords = scaffold_init_coords
            rebuilt_min_coords = scaffold_init_min_coords

        lig_rmsd = np.sqrt(np.mean(((lig_coords - lig_min_coords)**2).sum(-1)))
        h_rmsd = np.sqrt(np.mean(((h_coords - h_min_coords)**2).sum(-1)))
        pocket_rmsd = np.sqrt(np.mean(((pocket_coords - pocket_min_coords)**2).sum(-1)))
        scaffold_rmsd = np.sqrt(np.mean(((scaffold_coords - scaffold_min_coords)**2).sum(-1)))
        rebuilt_rmsd = np.sqrt(np.mean(((rebuilt_coords - rebuilt_min_coords)**2).sum(-1)))

        return {'basename': subdir, 'Ligand_RMSD': lig_rmsd, 'Pocket_RMSD': pocket_rmsd, 'Scaffold_RMSD': scaffold_rmsd, 'Rebuilt_RMSD': rebuilt_rmsd, 'H_RMSD': h_rmsd}

    except Exception as e:
        return {'basename': subdir, 'Ligand_RMSD': None, 'Pocket_RMSD': None, 'Scaffold_RMSD': None, 'Rebuilt_RMSD': None, 'H_RMSD': None}

if __name__ == '__main__':

    subdir_list = os.listdir(f'{DATA_DIR}/mol2_files')
    list_of_dicts = Parallel(n_jobs = 64, verbose = 10)(delayed(calculate_rmsd)(subdir) for subdir in subdir_list)

    df = pd.DataFrame(list_of_dicts)
    df.to_csv(f'{DATA_DIR}/metadata/CROWN_rmsd.csv', index = False, float_format = '%.4f')

