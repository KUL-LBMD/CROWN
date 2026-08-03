import os

from src.config import DATA_DIR
from src.core.download_mmcif import download_mmcif
from src.core.structure_fixer import fix_structures
from src.core.pli_filter import filter_structures

NUM_CORES = 96

### Step 1: Download raw files from rcsb ###

# Initialize empty subdirectories
new_subdirs = [f'{DATA_DIR}/mmCIF', f'{DATA_DIR}/mmCIF/raw', f'{DATA_DIR}/mmCIF/clean', f'{DATA_DIR}/mmCIF/ccd',
	f'{DATA_DIR}/pdb', f'{DATA_DIR}/pdb/raw', f'{DATA_DIR}/pdb/fixed', f'{DATA_DIR}/systems',
	f'{DATA_DIR}/processed_systems', f'{DATA_DIR}/complexes', f'{DATA_DIR}/mol2_files']

for subdir in new_subdirs:
	os.makedirs(subdir, exist_ok = True)

print('Running step 1: downloading mmCIF files and metadata')
download_mmcif()

### Step 2: Clean up structures and build PLI systems ###
fix_structures(num_cores = NUM_CORES)

### Step 3: Run PLI filter ###
print('Step 3: starting PLI filter')
filter_structures(num_cores = NUM_CORES)
