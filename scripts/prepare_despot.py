import shutil
import os
import pandas as pd

from src.config import DATA_DIR
DESPOT_DIR = '/media/drives/drive3/robin/DESPOT/data'

df = pd.read_parquet(f'{DATA_DIR}/metadata/CROWN_train.parquet')
basename_list = df['basename'].tolist()
num_files = len(basename_list)

for i, basename in enumerate(basename_list):
	shutil.copy(f'{DATA_DIR}/mol2_files/{basename}/receptor.mol2', f'{DESPOT_DIR}/CROWN_xtal/processed_mol2/receptor/{basename}.mol2')
	shutil.copy(f'{DATA_DIR}/mol2_files/{basename}/receptor_minimized.mol2', f'{DESPOT_DIR}/CROWN_train/processed_mol2/receptor/{basename}.mol2')
	shutil.copy(f'{DATA_DIR}/mol2_files/{basename}/ligand.mol2', f'{DESPOT_DIR}/CROWN_xtal/processed_mol2/ligand/{basename}.mol2')
	shutil.copy(f'{DATA_DIR}/mol2_files/{basename}/ligand_minimized.mol2', f'{DESPOT_DIR}/CROWN_train/processed_mol2/ligand/{basename}.mol2')
	print(f'{i} / {num_files}')
