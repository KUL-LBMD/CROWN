import pandas as pd
import os
import shutil
from joblib import Parallel, delayed

from src.config import DATA_DIR
from src.core.download_mmcif import download_mmcif
from src.core.get_pdb_metadata import get_pdb_metadata
from src.core.structure_fixer import fix_structures
from src.core.pli_filter import filter_structures
from src.core.system_builder import process_system
from src.core.refine_system import safe_refine_system
from src.core.mol2_converter import convert_system
from src.core.metadata_merger import merge_metadata

from src.utils.rmsd_calculator import calculate_rmsd
from src.utils.property_calculator import process_ligand
from src.utils.run_posebusters import run_posebusters
from src.utils.protein_metadata import get_protein_metadata

from src.cluster.cluster_plec import cluster_plec
from src.cluster.cluster_mmseqs import cluster_mmseqs
from src.cluster.cluster_ecfp4 import cluster_ecfp4
from src.cluster.cluster_merger import merge_clusters

NUM_CORES = 96

### Step 1: Download raw files from rcsb ###

# Initialize empty subdirectories
new_subdirs = [f'{DATA_DIR}/mmCIF', f'{DATA_DIR}/mmCIF/raw', f'{DATA_DIR}/mmCIF/clean', f'{DATA_DIR}/mmCIF/ccd',
	f'{DATA_DIR}/pdb', f'{DATA_DIR}/pdb/raw', f'{DATA_DIR}/pdb/fixed', f'{DATA_DIR}/systems',
	f'{DATA_DIR}/processed_systems', f'{DATA_DIR}/complexes', f'{DATA_DIR}/mol2_files']

for subdir in new_subdirs:
	os.makedirs(subdir, exist_ok = True)

#print('Running step 1: downloading mmCIF files and metadata')
#download_mmcif()

#for subdir in new_subdirs:
#	os.makedirs(subdir, exist_ok = True)

#get_pdb_metadata()
#get_protein_metadata()

### Step 2: Clean up structures and build PLI systems ###
#fix_structures(num_cores = NUM_CORES)

### Step 3: Run PLI filter ###
#print('Step 3: starting PLI filter')
#filter_structures(num_cores = NUM_CORES)

### Step 4: Prepare systems for energy minimization ###
#print('Step 4: Preparing energy minimization')
#df = pd.read_csv(f'{DATA_DIR}/metadata/pli_filter_pass.csv')
#basename_list = df['basename'].tolist()
#Parallel(n_jobs = NUM_CORES, verbose = 10)(delayed(process_system)(basename) for basename in basename_list)

### Step 5: Run energy minimization ###
print('Step 5: Starting energy minimization')
basename_list = os.listdir(f'{DATA_DIR}/systems')
Parallel(n_jobs = NUM_CORES, verbose = 10)(delayed(safe_refine_system)(basename) for basename in basename_list)

### Step 6: Post-processing: file conversion with OpenBabel ###
print('Step 6: file conversion')
basename_list = os.listdir(f'{DATA_DIR}/processed_systems')
Parallel(n_jobs = NUM_CORES, verbose = 10, backend = 'multiprocessing')(delayed(convert_system)(basename) for basename in basename_list) # Different backend, since most of the work here is OpenBabel, not Python

### Step 7: Metadata calculation ###
print('Step 7: metadata calculation')
basename_list = os.listdir(f'{DATA_DIR}/mol2_files')

# 7.1: RMSD
list_of_dicts = Parallel(n_jobs = NUM_CORES, verbose = 10)(delayed(calculate_rmsd)(basename) for basename in basename_list)
df = pd.DataFrame(list_of_dicts)
df.to_csv(f'{DATA_DIR}/metadata/CROWN_rmsd.csv', index = False, float_format = '%.4f')

# 7.2: Ligand properties
list_of_dicts = Parallel(n_jobs = NUM_CORES, verbose = 10)(delayed(process_ligand)(basename) for basename in basename_list)
df = pd.DataFrame(list_of_dicts)
df.to_csv(DATA_DIR / 'metadata' / 'CROWN_ligand_data.csv', index = False, float_format = '%.4f')

# 7.3: PoseBusters
results = Parallel(n_jobs = NUM_CORES, verbose = 10)(delayed(run_posebusters)(basename) for basename in basename_list)
flat_results = [x for sublist in results if sublist for x in sublist]
results_df = pd.concat(flat_results, ignore_index = True)
check_cols = [c for c in results_df.columns if results_df[c].dtype == bool]
results_df["n_failed_checks"] = (~results_df[check_cols]).sum(axis=1)
results_df["pb_valid"] = results_df[check_cols].all(axis=1)   # True only if every check passed
results_df.to_csv(f'{DATA_DIR}/metadata/posebusters.csv', index = False)

# 7.4: Merge all metadata
merge_metadata()

### Step 8: Hierarchical clustering ###
print('Step 8: clustering')
cluster_plec(num_cores = NUM_CORES)
#cluster_mmseqs()
cluster_ecfp4()
#merge_clusters()

### Step 9: Remove bad complexes ###
df = pd.read_parquet(f'{DATA_DIR}/metadata/CROWN_metadata.parquet')
basenames = set(df['basename'].tolist())
subdirs = set(os.listdir(f'{DATA_DIR}/complexes'))
subdirs_to_remove = subdirs - basenames
for subdir in subdirs_to_remove:
	shutil.rmtree(f'{DATA_DIR}/complexes/{subdir}')
	shutil.rmtree(f'{DATA_DIR}/mol2_files/{subdir}')
