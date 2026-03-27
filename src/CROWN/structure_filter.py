from src.config import DATA_DIR, SOURCE_DB_PATH

import pandas as pd
import os

from rdkit import Chem

NORMAL_ATOMS = {6, 7, 8, 16, 15, 9, 17, 53, 35}  # C, N, O, S, P, F, Cl, I, Br

def has_rare_atoms(smiles):
    """True if the molecule contains both normal and non-normal atoms."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False  # or True, depending on how you want to handle parse failures
    atomic_nums = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    has_normal = bool(atomic_nums & NORMAL_ATOMS)
    has_other = bool(atomic_nums - NORMAL_ATOMS)
    return has_normal and has_other

def filter_structures():
	"""
	Read PLInder metadata and select only relevant PLI systems

	Returns
	-------

	filtered_subset [pd.DataFrame]:
		- basename [str]: PLI system identifier
		- entry_resolution [float]: resolution of crystal structure
		- system_ligand_validation_average_rsr [float]: RSR of ligand
		- system_ligand_validation_average_rscc [float]: RSCC of ligand
		- system_pocket_UniProt [str]: UniProt ID of receptor
		- system_pocket_CATH [str]: CATH ID of receptor
		- ligand_unique_ccd_code [str]: CCD code of ligand
		- ligand_rdkit_canonical_smiles [str]: Canonical SMILES representation of ligand

	"""

	known_codes = {"HEM", "MGD", "SF4"}

	df = pd.read_parquet(SOURCE_DB_PATH / 'index' / 'annotation_table.parquet')
	df['basename'] = df['system_id'].str.lower() + '_' + df['ligand_instance_chain'].str.lower()
	mask_known = df["ligand_unique_ccd_code"].isin(known_codes)
	mask_rare_atoms = df["ligand_rdkit_canonical_smiles"].map(has_rare_atoms)
	df["is_rare"] = mask_rare_atoms & ~mask_known
	rare_system_ids = df.loc[df["is_rare"], "system_id"].unique()

	dir_list = os.listdir(SOURCE_DB_PATH / 'systems')

	subset = df[
		(df['ligand_num_unresolved_heavy_atoms'] == 0)
		& (df['entry_determination_method'] == 'X-RAY DIFFRACTION')
		& (df['entry_resolution'] <= 3.0)
		& (df['system_ligand_validation_average_rsr'] <= 0.3)
		& (df['system_ligand_validation_average_rscc'] >= 0.8)
		& (df['system_id'].isin(dir_list))
		].copy()

	print(len(df))
	print(len(subset))

	subset2 = subset[
		(subset['ligand_is_ion'] == False)
		& (subset['ligand_is_artifact'] == False)
		& (subset['ligand_is_covalent'] == False)
		].copy()

	# Rest stays the same
	df_clean = subset2[~subset2["system_id"].isin(rare_system_ids)].drop(columns="is_rare").copy()
	print(len(df_clean))

	columns_to_select = ['basename', 'system_id', 'ligand_instance_chain', 'entry_resolution', 'system_ligand_validation_average_rsr', 'system_ligand_validation_average_rscc',
		'system_pocket_UniProt', 'system_pocket_CATH', 'ligand_unique_ccd_code', 'ligand_rdkit_canonical_smiles', 'ligand_num_unresolved_heavy_atoms', 'ligand_is_covalent', 'ligand_is_artifact', 'ligand_is_ion', 'ligand_num_rot_bonds', 
		'ligand_num_hbd', 'ligand_num_hba', 'ligand_num_heavy_atoms']

	filtered_subset = df_clean[columns_to_select].copy()

	# Save metadata
	filtered_subset.to_csv(DATA_DIR / 'CROWN' / 'metadata' / 'structure_filter_pass.csv', index = False, float_format = '%.3f')

	return filtered_subset
