from src.config import DATA_DIR

import pandas as pd
import os
from posebusters import PoseBusters
from joblib import Parallel, delayed

def run(subdir):
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

if __name__ == "__main__":
	subdir_list = os.listdir(f'{DATA_DIR}/complexes')
	results = Parallel(n_jobs = 32, verbose = 10, backend = 'multiprocessing')(delayed(run)(subdir) for subdir in subdir_list)
	flat_results = [x for sublist in results if sublist for x in sublist]
	results_df = pd.concat(flat_results, ignore_index = True)
	check_cols = [c for c in results_df.columns if results_df[c].dtype == bool]
	results_df["n_failed_checks"] = (~results_df[check_cols]).sum(axis=1)
	results_df["pb_valid"] = results_df[check_cols].all(axis=1)   # True only if every check passed
	results_df.to_csv(f'{DATA_DIR}/metadata/posebusters.csv', index = False)
