from src.config import DATA_DIR
from src.CROWN.utils import remove_artifacts_and_fix_quotes

import numpy as np
import pandas as pd
import os
import gemmi
import freesasa
from scipy.spatial import KDTree
import re
from collections import defaultdict, Counter
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

# Rejection-stage labels (used for the summary). Order matters: this is the
# order in which checks are applied, and also the order printed in the summary.
REJECT_STAGES = [
	'parse_failed',
	'validation_unavailable',
	'validation_missing_residues',
	'rsr_rscc_threshold',
	'too_few_contacts',
	'low_sas_ratio',
	'redundant_pose',
]

### Helper functions ###
def _safe_mean(vals):
	return np.mean(vals) if len(vals) else np.nan

def _heavy_atom_count(res: gemmi.Residue) -> int:
	return sum(1 for atom in res if not atom.element.name in {'H', 'D'})

def get_validation_data(pdb_id: str):
	"""
	Get PDB validation data for RSR and RSCC

	Returns
	-------
	results: Dictionary
		- keys: (chain_id, res_name, res_num)
		- values: (rsr, rscc)
	Returns None when the validation report cannot be fetched/parsed.
	"""

	url = f"https://files.rcsb.org/pub/pdb/validation_reports/{pdb_id[1:3]}/{pdb_id}/{pdb_id}_validation.xml.gz"
	try:
		r = requests.get(url, timeout=30)
		r.raise_for_status()

	except requests.RequestException:
		return None

	try:
		xml_data = gzip.decompress(r.content)
		root = ET.fromstring(xml_data)
	except (OSError, ET.ParseError):
		return None

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

def calculate_rsr_rscc(basename: str, res_info: Dict[Tuple[str, str, str], Tuple[float, float]], structure: gemmi.Structure, ligand_coords: np.ndarray):
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

	ligand_keys = set()
	pocket_keys = set()
	chain_set = set()

	# --- Ligand: check within 0.1 tolerance
	ns = gemmi.NeighborSearch(model, structure.cell, SHELL_RADIUS).populate()
	for pos in ligand_coords:
		gemmi_pos = gemmi.Position(*pos)
		for mark in ns.find_atoms(gemmi_pos, '\0', radius = 0.1):
			cra = mark.to_cra(model)
			if _heavy_atom_count(cra.residue) <= 2:
				continue
			ligand_keys.add((cra.chain.name, cra.residue.name, str(cra.residue.seqid)))

	# --- Pocket: residues on ANY other chain within SHELL_RADIUS ---
	for pos in ligand_coords:
		gemmi_pos = gemmi.Position(*pos)
		for mark in ns.find_atoms(gemmi_pos, '\0', radius = SHELL_RADIUS):
			cra = mark.to_cra(model)
			if _heavy_atom_count(cra.residue) <= 2:
				continue
			pocket_keys.add((cra.chain.name, cra.residue.name, str(cra.residue.seqid)))
			chain_set.add(cra.chain.name)

	ligand_rsr = _safe_mean([res_info[x][0] if x in res_info else np.nan for x in ligand_keys])
	ligand_rscc = _safe_mean([res_info[x][1] if x in res_info else np.nan for x in ligand_keys])
	pocket_rsr = _safe_mean([res_info[x][0] if x in res_info else np.nan for x in pocket_keys])
	pocket_rscc = _safe_mean([res_info[x][1] if x in res_info else np.nan for x in pocket_keys])

	return ligand_rsr, ligand_rscc, pocket_rsr, pocket_rscc, chain_set, len(ligand_keys), len(pocket_keys)

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

def _make_rejection(filename: str, stage: str, detail: str, **metrics) -> Dict:
	"""Build a uniform rejection record. metrics get serialized as floats/ints."""
	rec = {
		'filename': filename[:-4] if filename.endswith('.pdb') else filename,
		'stage': stage,
		'detail': detail,
	}
	rec.update(metrics)
	return rec

def process_group(pdb_id: str, group):
	"""
	Function for joblib parallellization.

	Parameters
	----------
	pdb_id [str]
	group [List[str]]: List of filenames in group

	Returns
	-------
	results [List[Dict]] : entries that passed all filters
	rejections [List[Dict]] : per-file rejection records with stage + detail + metrics
	"""

	results = []
	rejections = []
	taken_arrays = []
	taken_filenames = []  # parallel to taken_arrays, for redundancy reporting

	# Group-level: load mmCIF and validation data once per pdb_id
	try:
		raw_structure = gemmi.read_structure(f'{DATA_DIR}/mmCIF/raw/{pdb_id}.cif')
		clean_structure = remove_artifacts_and_fix_quotes(raw_structure)
	except Exception as exc:
		# If the structure itself is unreadable, every file in the group fails.
		for filename in group:
			rejections.append(_make_rejection(
				filename, 'parse_failed',
				f'failed to read mmCIF for {pdb_id}: {type(exc).__name__}: {exc}'
			))
		return results, rejections

	res_info = get_validation_data(pdb_id)
	validation_available = isinstance(res_info, dict) and len(res_info) > 0

	for filename in group:

		file_path = f'{DATA_DIR}/pdb/fixed/{filename}'
		try:
			prot_coords, lig_coords, prot_radii, lig_radii, lig_name = parse_pdb(file_path)
		except Exception as exc:
			rejections.append(_make_rejection(
				filename, 'parse_failed',
				f'parse_pdb raised {type(exc).__name__}: {exc}'
			))
			continue

		if len(prot_coords.shape) != 2 or len(lig_coords.shape) != 2:
			rejections.append(_make_rejection(
				filename, 'parse_failed',
				'empty protein or ligand coordinate array',
				n_prot_atoms=int(prot_coords.size // 3) if prot_coords.size else 0,
				n_lig_atoms=int(lig_coords.size // 3) if lig_coords.size else 0,
			))
			continue

		# Group-level validation gate: if the report was unavailable, skip
		# but tag the rejection clearly so it's distinguishable from
		# "report present but residue missing from it".
		if not validation_available:
			rejections.append(_make_rejection(
				filename, 'validation_unavailable',
				f'validation report for {pdb_id} could not be fetched/parsed',
				lig_name=lig_name,
			))
			continue

		# Check 1: RSR and RSCC
		ligand_rsr, ligand_rscc, pocket_rsr, pocket_rscc, chain_set, n_lig_keys, n_pocket_keys = \
			calculate_rsr_rscc(filename, res_info, clean_structure, lig_coords)

		if np.isnan([ligand_rsr, ligand_rscc, pocket_rsr, pocket_rscc]).any():
			# Distinguish the two failure modes for the residue-key lookup:
			# either no keys were found at all, or none of the keys were in
			# the validation report.
			if n_lig_keys == 0 or n_pocket_keys == 0:
				detail = f'no ligand_keys ({n_lig_keys}) or pocket_keys ({n_pocket_keys}) identified'
			else:
				detail = ('validation report present, but residue keys absent from it '
				          '(NaN RSR/RSCC after lookup)')
			rejections.append(_make_rejection(
				filename, 'validation_missing_residues', detail,
				lig_name=lig_name,
				ligand_rsr=ligand_rsr, ligand_rscc=ligand_rscc,
				pocket_rsr=pocket_rsr, pocket_rscc=pocket_rscc,
				n_ligand_keys=n_lig_keys, n_pocket_keys=n_pocket_keys,
			))
			continue

		if ligand_rsr > 0.3 or pocket_rsr > 0.3 or ligand_rscc < 0.8 or pocket_rscc < 0.8:
			# Build a precise reason listing which thresholds were violated
			fails = []
			if ligand_rsr > 0.3:
				fails.append(f'ligand_rsr={ligand_rsr:.3f} > 0.3')
			if pocket_rsr > 0.3:
				fails.append(f'pocket_rsr={pocket_rsr:.3f} > 0.3')
			if ligand_rscc < 0.8:
				fails.append(f'ligand_rscc={ligand_rscc:.3f} < 0.8')
			if pocket_rscc < 0.8:
				fails.append(f'pocket_rscc={pocket_rscc:.3f} < 0.8')
			rejections.append(_make_rejection(
				filename, 'rsr_rscc_threshold',
				'; '.join(fails),
				lig_name=lig_name,
				ligand_rsr=ligand_rsr, ligand_rscc=ligand_rscc,
				pocket_rsr=pocket_rsr, pocket_rscc=pocket_rscc,
			))
			continue

		# Check 2: more than 10 close contacts?
		lig_tree = KDTree(lig_coords)
		close_contacts = lig_tree.query_ball_point(prot_coords, r=CONTACT_RADIUS)
		num_contacts = sum(len(x) for x in close_contacts)

		if num_contacts < 10:
			rejections.append(_make_rejection(
				filename, 'too_few_contacts',
				f'num_contacts={num_contacts} < 10 (within {CONTACT_RADIUS} Å)',
				lig_name=lig_name,
				num_contacts=num_contacts,
				ligand_rsr=ligand_rsr, ligand_rscc=ligand_rscc,
				pocket_rsr=pocket_rsr, pocket_rscc=pocket_rscc,
			))
			continue

		# Check 3: Delta-SAS ratio
		sas_ratio = calculate_delta_sas(prot_coords, lig_coords, prot_radii, lig_radii)
		if sas_ratio < 0.4:
			rejections.append(_make_rejection(
				filename, 'low_sas_ratio',
				f'sas_ratio={sas_ratio:.3f} < 0.4',
				lig_name=lig_name,
				sas_ratio=sas_ratio, num_contacts=num_contacts,
				ligand_rsr=ligand_rsr, ligand_rscc=ligand_rscc,
				pocket_rsr=pocket_rsr, pocket_rscc=pocket_rscc,
			))
			continue

		# Check 4: rigid-body alignment, greedy pruning
		idx_6 = lig_tree.query_ball_point(prot_coords, r=6)
		mask_6 = np.fromiter((len(x) > 0 for x in idx_6), dtype=bool)
		subset = prot_coords[mask_6]
		current_arr = np.concatenate((subset, lig_coords), axis=0)

		redundant_against = None
		redundant_maxdev = None
		for previous_arr, previous_name in zip(taken_arrays, taken_filenames):
			maxdev = icp(current_arr, previous_arr)
			if maxdev < MAXDEV_THRESHOLD:
				redundant_against = previous_name
				redundant_maxdev = maxdev
				break

		if redundant_against is not None:
			rejections.append(_make_rejection(
				filename, 'redundant_pose',
				f'maxdev={redundant_maxdev:.4f} < {MAXDEV_THRESHOLD} vs {redundant_against}',
				lig_name=lig_name,
				maxdev=redundant_maxdev,
				redundant_against=redundant_against,
				sas_ratio=sas_ratio, num_contacts=num_contacts,
				ligand_rsr=ligand_rsr, ligand_rscc=ligand_rscc,
				pocket_rsr=pocket_rsr, pocket_rscc=pocket_rscc,
			))
			continue

		# Survived everything → keep
		taken_arrays.append(current_arr)
		taken_filenames.append(filename[:-4])
		entry_results = {
			'filename': filename[:-4],
			'lig_name': lig_name,
			'sas_ratio': sas_ratio,
			'ligand_rsr': ligand_rsr, 'ligand_rscc': ligand_rscc,
			'pocket_rsr': pocket_rsr, 'pocket_rscc': pocket_rscc,
			'chain_set': '-'.join(chain_set),
		}
		results.append(entry_results)

	return results, rejections


def _print_summary(total_seen: int, n_pass: int, stage_counts: Counter):
	"""Print a tidy stage-by-stage rejection summary."""
	print('\n' + '=' * 60)
	print(f'PLI filter summary  ({total_seen} files seen)')
	print('=' * 60)
	header = f'  {"stage":<32}{"count":>8}{"% of total":>12}'
	print(header)
	print('  ' + '-' * (len(header) - 2))
	total_rejected = 0
	for stage in REJECT_STAGES:
		c = stage_counts.get(stage, 0)
		total_rejected += c
		pct = (c / total_seen * 100) if total_seen else 0.0
		print(f'  {stage:<32}{c:>8}{pct:>11.2f}%')
	print('  ' + '-' * (len(header) - 2))
	pct_rej = (total_rejected / total_seen * 100) if total_seen else 0.0
	pct_pass = (n_pass / total_seen * 100) if total_seen else 0.0
	print(f'  {"TOTAL rejected":<32}{total_rejected:>8}{pct_rej:>11.2f}%')
	print(f'  {"TOTAL passed":<32}{n_pass:>8}{pct_pass:>11.2f}%')
	print('=' * 60 + '\n')


def main():
	"""
	Prune database based on 4 criteria:
	1. RSR and RSCC of ligand and pocket residues
	2. More than 10 close contacts between ligand and receptor
	3. Delta-SAS ratio between free and bound ligand > 0.4
	4. Remove redundant structures from same PDB ID through rigid-body alignment

	This verbose variant additionally:
	- writes a per-file rejection log to {DATA_DIR}/metadata/pli_filter_rejected.csv
	  with the stage, a human-readable detail string, and the relevant numeric
	  metrics for each thrown-out file.
	- prints a stage-by-stage summary to stdout.
	"""

	groups = defaultdict(list)
	for filename in os.listdir(f'{DATA_DIR}/pdb/fixed'):
		pdb_id = filename[:4]
		groups[pdb_id].append(filename)

	total_seen = sum(len(v) for v in groups.values())
	print(f'[pli_filter] Found {total_seen} files in {len(groups)} PDB groups.')

	worker_outputs = Parallel(n_jobs=64, verbose=10)(
		delayed(process_group)(pdb_id, group) for pdb_id, group in groups.items()
	)

	flat_results = [x for results, _ in worker_outputs for x in results]
	flat_rejections = [r for _, rejections in worker_outputs for r in rejections]

	# --- Pass-list CSV (unchanged contract from the original script) ---
	df_pass = pd.DataFrame(flat_results)
	pass_path = f'{DATA_DIR}/metadata/pli_filter_pass.csv'
	df_pass.to_csv(pass_path, index=False, float_format='%.6f')
	print(f'[pli_filter] Wrote {len(df_pass)} passing entries → {pass_path}')

	# --- Rejection-list CSV ---
	df_rej = pd.DataFrame(flat_rejections)
	# Order columns: identifying info first, then metrics. Any metric column not
	# present in a row will be left blank by pandas, which is what we want.
	preferred_order = [
		'filename', 'stage', 'detail', 'lig_name',
		'ligand_rsr', 'ligand_rscc', 'pocket_rsr', 'pocket_rscc',
		'n_ligand_keys', 'n_pocket_keys',
		'num_contacts', 'sas_ratio',
		'maxdev', 'redundant_against',
		'n_prot_atoms', 'n_lig_atoms',
	]
	cols = [c for c in preferred_order if c in df_rej.columns]
	cols += [c for c in df_rej.columns if c not in cols]
	if cols:
		df_rej = df_rej[cols]
	rej_path = f'{DATA_DIR}/metadata/pli_filter_rejected.csv'
	df_rej.to_csv(rej_path, index=False, float_format='%.6f')
	print(f'[pli_filter] Wrote {len(df_rej)} rejection entries → {rej_path}')

	# --- Stage-by-stage summary ---
	stage_counts = Counter(r['stage'] for r in flat_rejections)
	_print_summary(total_seen, len(flat_results), stage_counts)


if __name__ == '__main__':
	main()
