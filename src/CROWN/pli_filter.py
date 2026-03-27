from src.config import SOURCE_DB_PATH, DATA_DIR

import os
import shutil
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import re

LIG_LIST = ['LIG', 'HEM', 'MGD']

### Helper functions ###
def icp(mobile, target, max_iters = 100, tolerance = 1e-6):

	"""
	Algorithm for rigid-body alignment of point clouds
	(equal number of points not required)

	Parameters
	----------

	mobile [L1, 3]
	target [L2, 3]

	Returns
	-------
	max_dev [float]: Maximum deviation between two atoms after alignment
	"""

	if mobile.shape[0] > target.shape[0]:
		target, mobile = mobile, target

	tree = KDTree(target)

	prev_error = np.inf
	R_total = np.eye(3)
	t_total = np.zeros(3)

	for i in range(max_iters):
		dists, idx = tree.query(mobile)
		Q = target[idx]
		R, t = kabsch(mobile, Q)
		R_total = R @ R_total

		mobile = (R @ mobile.T).T + t

		mean_error = np.mean(dists)
		if abs(prev_error - mean_error) < tolerance:
			break
		prev_error = mean_error

	# Compute RMSD
	deviation = Q - mobile
	square_dev = np.sum(deviation**2, axis = 1)
	max_dev = np.max(np.sqrt(square_dev))

	return max_dev

def kabsch(mobile, target):
	"""
	Align mobile ligand to target using Kabsch algorithm

	Parameters
	----------
	mobile [L, 3]
	target [L, 3]

	Returns
	-------
	rmsd [float]
	"""

	mobile_center = np.mean(mobile, axis = 0)
	target_center = np.mean(target, axis = 0)

	mobile -= mobile_center
	target -= target_center

	H = mobile.T @ target
	U, S, Vt = np.linalg.svd(H)
	rotation = Vt.T @ U.T

	# Ensure right-handed coordinate system
	if np.linalg.det(rotation) < 0:
		Vt[-1,:] *= -1
		rotation = Vt.T @ U.T

	t = target.mean(axis=0) - rotation @ mobile.mean(axis=0)

	return rotation, t

def parse_pdb(file_path):
	"""
	Parses protein and ligand coordinates from PDB file

	Parameters
	----------

	file_path [str]: full path to PDB file

	Returns
	-------

	prot_coords [np.array(L1, 3)]
	lig_coords [np.array(L2, 3)]
	"""

	prot_coords_list = []
	lig_coords_list = []

	with open(file_path, 'r') as f:
		for line in f:
			if line.startswith(('HETATM', 'ATOM')):
				line = line.strip()
				res_name = line[17:20].strip()
				chain_id = line[21].strip()
				atom_name = line[12:16].strip()
				element = line[76:78].strip()
				if element in {'H', 'D'}:
					continue

				try:
					x = float(line[30:38])
					y = float(line[38:46])
					z = float(line[46:54])
				except ValueError:
					# Fallback for funky parsing
					coord_str = line[30:].strip()
					numbers = re.findall(r'[-+]?\d*\.\d+|\d+', coord_str)
					x, y, z = map(float, numbers[:3])

				if res_name in LIG_LIST:
					lig_coords_list.append([x,y,z])

				else:
					prot_coords_list.append([x,y,z])

	prot_coords = np.array(prot_coords_list)
	lig_coords = np.array(lig_coords_list)

	return prot_coords, lig_coords

class PLI_Filter:
	def __init__(self, plinder_subset):
		"""
		Parameters
		----------

		plinder_subset [pd.DataFrame]
			- basename [str]: PLI system identifier
			- system_id [str]: input filename
			- ligand_instance_chain [str]: ligand chain identifier
			- entry_resolution [float]: resolution of crystal structure
			- system_ligand_validation_average_rsr [float]: RSR of ligand
			- system_ligand_validation_average_rscc [float]: RSCC of ligand
			- system_pocket_UniProt [str]: UniProt ID of receptor
			- system_pocket_CATH [str]: CATH ID of receptor
			- ligand_unique_ccd_code [str]: CCD code of ligand
			- ligand_rdkit_canonical_smiles [str]: Canonical SMILES representation of ligand
		"""

		self.df = plinder_subset
		self.df['pdb_id'] = self.df['basename'].str[:4]

	def prune_maxdev(self, maxdev_threshold = 0.1):

		"""
		Prune initial dataset based on overlapping structures.

		Parameters
		----------
		maxdev_threshold [float]: Threshold for maxdev between aligned atoms.
			If maxdev below this threshold, the file will be discarded from the dataset.

		Returns
		-------
		maxdev_subset [pd.DataFrame]
			- Same as plinder_subset, but fewer rows
		"""

		basenames_to_keep = set()
		num_groups = len(self.df['pdb_id'].unique())

		total_entries = 0
		removed_file_missing = 0
		removed_parse_error = 0
		removed_contacts_atoms = 0
		removed_maxdev = 0

		for i, (group, rows) in enumerate(self.df.groupby('pdb_id', sort = False)):
			taken_arrays = []

			print(f'{i} / {num_groups} done')

			for row in rows.itertuples(index = False):
				total_entries += 1
				basename = row.basename
				file_path = f'{DATA_DIR}/CROWN/raw_pdb/{basename}.pdb'
				if not os.path.isfile(file_path):
					removed_file_missing += 1
					continue

				prot_coords, lig_coords = parse_pdb(file_path)

				if len(prot_coords.shape) != 2 or len(lig_coords.shape) != 2:
					removed_parse_error += 1
					continue

				lig_tree = KDTree(lig_coords)
				# Pairs within 4 A
				idx_4 = lig_tree.query_ball_point(prot_coords, r = 4)
				num_contacts = sum(len(x) for x in idx_4)

				if num_contacts < 10 or lig_coords.shape[0] < 10 or lig_coords.shape[0] > 100:
					removed_contacts_atoms += 1
					continue

				# Protein atoms within 6A. Construct local pocket environment for ICP rigid-body alignment
				idx_6 = lig_tree.query_ball_point(prot_coords, r = 6)
				mask_6 = np.fromiter((len(x) > 0 for x in idx_6), dtype=bool)
				subset = prot_coords[mask_6]

				current_arr = np.concatenate((subset, lig_coords), axis = 0)

				redundant = False
				for previous_arr in taken_arrays:
					maxdev = icp(current_arr, previous_arr)
					if maxdev < maxdev_threshold:
						redundant = True
						break

				if not redundant:
					basenames_to_keep.add(basename)
					taken_arrays.append(current_arr)
				else:
					removed_maxdev += 1

		# Report
		kept = len(basenames_to_keep)
		report_path = DATA_DIR / 'CROWN' / 'metadata' / 'prune_maxdev_report.txt'
		with open(report_path, 'w') as f:
			f.write('=' * 60 + '\n')
			f.write('prune_maxdev report\n')
			f.write('=' * 60 + '\n')
			f.write(f'{"Total entries processed:":<40} {total_entries}\n')
			f.write(f'{"Removed (file missing):":<40} {removed_file_missing}\n')
			f.write(f'{"Removed (parse error):":<40} {removed_parse_error}\n')
			f.write(f'{"Removed (contacts/atom count):":<40} {removed_contacts_atoms}\n')
			f.write(f'{"Removed (maxdev < " + str(maxdev_threshold) + "):":<40} {removed_maxdev}\n')
			f.write(f'{"Entries kept:":<40} {kept}\n')
			f.write('=' * 60 + '\n')
		print(f'Report saved to {report_path}')

		# Update dataframe
		self.df.set_index('basename', drop = False, inplace = True)
		maxdev_subset = self.df.loc[self.df.index.isin(basenames_to_keep)]
		return maxdev_subset

	def wrapper(self, maxdev_threshold = 0.1, max_count = 500):
		maxdev_subset = self.prune_maxdev(maxdev_threshold = maxdev_threshold)
		maxdev_subset.to_csv(DATA_DIR / 'CROWN' / 'metadata' / 'pli_filter_pass.csv', index = False, float_format = '%.3f')
