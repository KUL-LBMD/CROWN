from src.config import DATA_DIR
from src.CROWN.utils import remove_artifacts_and_fix_quotes

import numpy as np
import pandas as pd
import os
import gemmi
import freesasa
from scipy.spatial import KDTree
import re
from collections import defaultdict
import requests
import xml.etree.ElementTree as ET
import gzip
from typing import List, Dict, Tuple

from joblib import Parallel, delayed

VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
    'S': 1.80, 'P': 1.80, 'F': 1.47, 'CL': 1.75,
    'BR': 1.85, 'I':  1.98, 'FE': 1.80, 'ZN': 1.39,
    'MG': 1.73, 'CA': 1.97, 'MN': 1.61, 'CU': 1.40,
}

MAXDEV_THRESHOLD = 0.1
CONTACT_RADIUS = 4
SHELL_RADIUS = 6

### Helper functions ###
def _heavy_atom_count(res: gemmi.Residue) -> int:
    return sum(1 for atom in res if not atom.element.name in {'H', 'D'})

def get_validation_data(pdb_id: str) -> Dict[Tuple[str, str, str], Tuple[float, float]]:
	"""
	Get PDB validation data for RSR and RSCC

	Returns
	-------
	results: Dictionary
		- keys: (chain_id, res_name, res_num)
		- values: (rsr, rscc)
	"""

	url = f"https://files.rcsb.org/pub/pdb/validation_reports/{pdb_id[1:3]}/{pdb_id}/{pdb_id}_validation.xml.gz"
	try:
		r = requests.get(url, timeout=30)
		r.raise_for_status()
		
	except requests.RequestException:
		return None, None

	xml_data = gzip.decompress(r.content)
	root = ET.fromstring(xml_data)
	namespace = root.tag.split("}")[0] + "}" if "}" in root.tag else ""

	results = {}

	for res in root.iter(f"{namespace}ModelledSubgroup"):
		chain_id = res.get('said')
		res_name = res.get('resname')
		res_num = res.get('resnum')
		rsr = res.get('rsr')
		rscc = res.get('rscc')

		if res_name != 'HOH' and rsr is not None and rscc is not None:
			results[(chain_id, res_name, res_num)] = (float(rsr), float(rscc))

	return results

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
	prot_radii = []
	lig_coords_list = []
	lig_radii = []
	lig_name_list = []

	previous_key = (None, None, None) # chain,res,num

	with open(file_path, 'r') as f:
		for line in f:
			if line.startswith(('HETATM', 'ATOM')):
				line = line.strip()
				res_name = line[17:20].strip()
				chain_id = line[21].strip()
				resnum = line[22:25].strip()
				atom_name = line[12:16].strip()
				element = line[76:78].strip().upper()
				if element in {'H', 'D'}:
					continue

				radius = VDW_RADII.get(element, 1.70)

				try:
					x = float(line[30:38])
					y = float(line[38:46])
					z = float(line[46:54])
				except ValueError:
					# Fallback for funky parsing
					coord_str = line[30:].strip()
					numbers = re.findall(r'[-+]?\d*\.\d+|\d+', coord_str)
					x, y, z = map(float, numbers[:3])

				if chain_id == 'A':
					lig_coords_list.append([x,y,z])
					lig_radii.append(radius)

					current_key = (chain_id, res_name, resnum)
					if current_key != previous_key:
						lig_name_list.append(res_name)
				else:
					prot_coords_list.append([x,y,z])
					prot_radii.append(radius)

				previous_key = current_key

	prot_coords = np.array(prot_coords_list)
	lig_coords = np.array(lig_coords_list)
	lig_name = '-'.join(lig_name_list)

	return prot_coords, lig_coords, prot_radii, lig_radii, lig_name

### Core functions
#-----------------

def calculate_rsr_rscc(basename: str, res_info: Dict[Tuple[str, str, str], Tuple[float, float]], structure: gemmi.Structure, chain_id: str):
	"""
	Calculate mean RSR and RSCC of given residues

	Parameters
	----------
	res_info: rsr and rscc for every (chain_id, res_name, res_num) key

	Returns
	-------
	mean_rsr [float]
	mean_rscc [float]
	"""

	model = structure[0]

	# Step 1: get (chain_id, res_name, res_num) keys for ligand and pocket
	# --- Ligand: all residues in chain_id with >2 heavy atoms ---
	ligand_keys = set()
	ligand_coords = []

	for chain in model:
		if chain.name != chain_id:
			continue
		for res in chain:
			if _heavy_atom_count(res) <= 2:
				continue
			key = (chain_id, res.name, str(res.seqid))
			ligand_keys.add(key)
			for atom in res:
				if not atom.is_hydrogen():
					ligand_coords.append(atom.pos)

	if not ligand_coords:
		print(f'{basename} - No ligand coords')
		empty_set = set()
		return np.nan, np.nan, np.nan, np.nan, empty_set
	
	# --- Pocket: residues on ANY other chain within SHELL_RADIUS ---
	ns = gemmi.NeighborSearch(model, structure.cell, SHELL_RADIUS).populate()
	pocket_keys = set()
	chain_set = set()

	for pos in ligand_coords:
		for mark in ns.find_atoms(pos, '\0', radius = SHELL_RADIUS):
			cra = mark.to_cra(model)
			if cra.chain.name == chain_id:
				continue  # skip ligand chain itself
			if _heavy_atom_count(cra.residue) <= 2:
				continue
			pocket_keys.add((cra.chain.name, cra.residue.name, str(cra.residue.seqid)))
			chain_set.add(cra.chain.name)

	ligand_rsr = np.mean([res_info[x][0] if x in res_info else np.nan for x in ligand_keys])
	ligand_rscc = np.mean([res_info[x][1] if x in res_info else np.nan for x in ligand_keys])
	pocket_rsr = np.mean([res_info[x][0] if x in res_info else np.nan for x in pocket_keys])
	pocket_rscc = np.mean([res_info[x][1] if x in res_info else np.nan for x in pocket_keys])

	return ligand_rsr, ligand_rscc, pocket_rsr, pocket_rscc, chain_set

def calculate_delta_sas(prot_coords, lig_coords, prot_radii, lig_radii):
	"""
	Calculates SASA difference between ligand in vacuum and ligand in receptor
	"""

	n_prot = len(prot_radii)
	n_lig = len(lig_radii)

	# Free ligand
	sasa_free_sum = freesasa.calcCoord(lig_coords.flatten(), lig_radii).totalArea()

	# Ligand in complex
	complex_coords = np.concatenate([prot_coords, lig_coords], axis = 0)
	complex_radii = np.concatenate([prot_radii, lig_radii])
	sasa_bound = freesasa.calcCoord(complex_coords.flatten(), complex_radii)
	sasa_bound_sum = sum(sasa_bound.atomArea(i) for i in range(n_prot, n_prot + n_lig))

	# Calculate delta-sas ratio
	delta_sasa = sasa_free_sum - sasa_bound_sum
	sasa_ratio = delta_sasa / sasa_free_sum

	return sasa_ratio

def process_group(pdb_id: str, group):
	"""
	Function for joblib parallellization.
	
	Parameters
	----------
	pdb_id [str]
	group [List[str]]: List of filenames in group
	
	Returns
	-------
	results [Dict]
        - filename
		- ligand_name
		- ligand_rsr
		- ligand_rscc
		- pocket_rsr
		- pocket_rscc
		- sas_ratio
	"""
	
	results = []
	taken_arrays = []
	raw_structure = gemmi.read_structure(f'{DATA_DIR}/mmCIF/raw/{pdb_id}.cif')
	clean_structure = remove_artifacts_and_fix_quotes(raw_structure)
	res_info = get_validation_data(pdb_id)

	for filename in group:

		# Check 1: RSR and RSCC
		chain_id = filename.split('.')[0].split('_')[1]
		ligand_rsr, ligand_rscc, pocket_rsr, pocket_rscc, chain_set = calculate_rsr_rscc(filename, res_info, clean_structure, chain_id)

		if np.isnan([ligand_rsr, ligand_rscc, pocket_rsr, pocket_rscc]).any():
			continue

		if ligand_rsr > 0.3 or pocket_rsr > 0.3 or ligand_rscc < 0.8 or pocket_rscc < 0.8:
			continue

		file_path = f'{DATA_DIR}/pdb/fixed/{filename}'
		prot_coords, lig_coords, prot_radii, lig_radii, lig_name = parse_pdb(file_path)

		if len(prot_coords.shape) != 2 or len(lig_coords.shape) != 2:
			print(f'Please check {filename}')
			continue

		# Check 2: more than 10 close contacts?
		lig_tree = KDTree(lig_coords)
		close_contacts = lig_tree.query_ball_point(prot_coords, r = CONTACT_RADIUS)
		num_contacts = sum(len(x) for x in close_contacts)

		if num_contacts < 10:
			continue

		# Check 3: Delta-SAS ratio
		sas_ratio = calculate_delta_sas(prot_coords, lig_coords, prot_radii, lig_radii)
		if np.isnan(sas_ratio):
			print(f'{filename} - SAS {sas_ratio}')
		if sas_ratio < 0.4:
			continue

		# Check 4: rigid-body alignment, greedy pruning
		# Protein atoms within 6A. Construct local pocket environment for ICP rigid-body alignment
		idx_6 = lig_tree.query_ball_point(prot_coords, r = 6)
		mask_6 = np.fromiter((len(x) > 0 for x in idx_6), dtype=bool)
		subset = prot_coords[mask_6]
		current_arr = np.concatenate((subset, lig_coords), axis = 0)

		redundant = False
		for previous_arr in taken_arrays:
			maxdev = icp(current_arr, previous_arr)
			if maxdev < MAXDEV_THRESHOLD:
				redundant = True
				break

		if not redundant:
			entry_results = {'filename': filename[:-4], 'lig_name': lig_name, 'sas_ratio': sas_ratio, 'ligand_rsr': ligand_rsr, 'ligand_rscc': ligand_rscc,
					'pocket_rsr': pocket_rsr, 'pocket_rscc': pocket_rscc, 'chain_set': '-'.join(chain_set)}
			results.append(entry_results)

	return results

def main():
	"""
	Prune database based on 3 criteria:
	1. RSR and RSCC of ligand and pocket residues
	2. More than 10 close contacts between ligand and receptor
	3. Delta-SAS ratio between free and bound ligand > 0.4
	4. Remove redundant structures from same PDB ID through rigid-body alignment
	"""

	results = []

	groups = defaultdict(list)
	for filename in os.listdir(f'{DATA_DIR}/pdb/fixed'):
		pdb_id = filename[:4]
		groups[pdb_id].append(filename)

	results = Parallel(n_jobs = 64, verbose = 10)(delayed(process_group)(pdb_id, group) for pdb_id, group in groups.items())
	flat_results = [x for sublist in results for x in sublist]

	# Report: store saved files and SAS ratios in a csv
	df = pd.DataFrame(flat_results)
	df.to_csv(f'{DATA_DIR}/metadata/pli_filter_pass.csv', index = False, float_format = '%.6f')

if __name__ == '__main__':
	main()
