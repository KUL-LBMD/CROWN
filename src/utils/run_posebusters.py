from src.config import DATA_DIR

import pandas as pd
import os
from posebusters import PoseBusters
from joblib import Parallel, delayed

def run_posebusters(subdir):
	# "dock" = ligand conditioned on protein, no ground-truth pose.
	# Use "redock" with mol_true if you also want RMSD/identity checks.
	buster = PoseBusters(config="dock")
	states = {
		"raw":       ("ligand.sdf",           "receptor.pdb"),
		"minimized": ("ligand_minimized.sdf", "receptor_minimized.pdb"),
	}

	df_list = []

	for state, (lig_name, rec_name) in states.items():
		try:
			df = buster.bust(mol_pred = f'{DATA_DIR}/complexes/{subdir}/{lig_name}',
					mol_cond = f'{DATA_DIR}/complexes/{subdir}/{rec_name}',
					full_report = False
			)

			df = df.reset_index()                 # flatten PoseBusters' MultiIndex into columns
			df.insert(0, "entry_id", subdir)  # the subdirectory name
			df.insert(1, "state", state)          # "raw" or "minimized"
			df_list.append(df)

		except Exception as e:
			print(f"[fail] {subdir} ({state}): {e}")
			return None

	return df_list
