import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from joblib import Parallel, delayed

from src.config import DATA_DIR

def process_ligand(subdir):
	"""
	Return a dict of descriptors for a single RDKit mol object.

	Parameters
	----------

	subdir [str]: CROWN system identifier

	Returns
	-------

	results [Dict]:
		- MW [float]
		- HeavyAtoms [int]
		- N+O_Atoms [int]
		- HBD [int]
		- HBA [int]
		- RotatableBonds [int]
		- NumRings [int]
		- TPSA [float]
		- QED [float]
		- SMILES [str]
		- MurckoScaffold [str]
	"""

	results = {'basename': subdir, 'MW': None, 'HeavyAtoms': None, 'N+O_Atoms': None, 'HBD': None, 'HBA': None, 'RotatableBonds': None, 'NumRings': None, 'TPSA': None, 'QED': None, 'SMILES': None, 'MurckoScaffold': None}
	file_path = f'{DATA_DIR}/systems/{subdir}/chain_A.sdf'

	if os.path.isfile(file_path):
		mol = next(Chem.SDMolSupplier(file_path))
		if mol is not None:
			results['MW'] = Descriptors.MolWt(mol)
			results['HeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
			results['N+O_Atoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in (7, 8))
			results['HBD'] = rdMolDescriptors.CalcNumHBD(mol)
			results['HBA'] = rdMolDescriptors.CalcNumHBA(mol)
			results['RotatableBonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
			results['NumRings'] = sum(1 for ring in mol.GetRingInfo().AtomRings())
			results['TPSA'] = Descriptors.TPSA(mol)
			results['QED'] = QED.qed(mol)
			results['SMILES'] = Chem.MolToSmiles(mol)

			try:
				scaffold = MurckoScaffold.GetScaffoldForMol(mol)
				results['MurckoScaffold'] = Chem.MolToSmiles(scaffold)

			except Exception:
				pass

		else:
			print(f'{file_path} failed parsing!')

	else:
		print(f'{file_path} not found!')

	return results

if __name__ == '__main__':
	#subdir_list = os.listdir(DATA_DIR / 'complexes')
	subdir_list = ['3fiv_D']
	list_of_dicts = Parallel(n_jobs = 1, verbose = 10)(delayed(process_ligand)(subdir) for subdir in subdir_list)
	df = pd.DataFrame(list_of_dicts)
	df.to_csv(DATA_DIR / 'metadata' / 'CROWN_ligand_data_new.csv', index = False, float_format = '%.4f')
