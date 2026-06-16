import pandas as pd
import os
import shutil

from src.config import DATA_DIR

df = pd.read_parquet(f'{DATA_DIR}/metadata/CROWN_full_annotated.parquet')
basenames = set(df['basename'].tolist())

subdirs = set(os.listdir(f'{DATA_DIR}/complexes'))
subdirs_to_remove = subdirs - basenames

for subdir in subdirs_to_remove:
	shutil.rmtree(f'{DATA_DIR}/complexes/{subdir}')
	shutil.rmtree(f'{DATA_DIR}/mol2_files/{subdir}')
