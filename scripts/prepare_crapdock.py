import shutil
import pandas as pd

from src.config import DATA_DIR

df = pd.read_csv(f'{DATA_DIR}/metadata/crapdock.csv')
basename_list = df['basename'].tolist()

for basename in basename_list:
	shutil.copy(f'{DATA_DIR}/complexes/{basename}/receptor_minimized.pdb', f'{DATA_DIR}/crapdock/receptor/{basename}.pdb')
	shutil.copy(f'{DATA_DIR}/complexes/{basename}/ligand_minimized.sdf', f'{DATA_DIR}/crapdock/ligand/{basename}.sdf')
