import pandas as pd

from src.config import DATA_DIR

def to_set(s):
	out = set()
	for v in s.dropna():
		if isinstance(v, list):
			out.update(x for x in v if x != '')
		else:
			out.add(v)
	return out

def merge_metadata():

	### Step 1: Load all metadata files ###
	ligand_df = pd.read_csv(f'{DATA_DIR}/metadata/CROWN_ligand_data.csv')
	pdb_df = pd.read_csv(f'{DATA_DIR}/metadata/pdb_metadata.csv')
	uniprot_df = pd.read_csv(f'{DATA_DIR}/metadata/uniprot_metadata.csv')
	pli_df = pd.read_csv(f'{DATA_DIR}/metadata/pli_filter_pass.csv')
	special_residues = pd.read_csv(f'{DATA_DIR}/metadata/special_residues.csv')
	rmsd_df = pd.read_csv(f'{DATA_DIR}/metadata/CROWN_rmsd.csv')

	### Step 2: Start merging: ligand_df serves as main "key" ###
	df = ligand_df.merge(pli_df, how = 'inner', on = ['basename'])
	df = df.merge(pdb_df, how = 'inner', on = ['pdb_id'])

	## 2.1: Update missing ligand info with special residues
	df = df.merge(special_residues, how = 'left', on = ['lig_name'], suffixes=("", "_new"))
	for col_name in ['MW', 'HeavyAtoms', 'N+O_Atoms', 'HBD', 'HBA', 'RotatableBonds', 'NumRings', 'TPSA', 'QED', 'SMILES', 'MurckoScaffold']:
		df[col_name] = df[f'{col_name}_new'].combine_first(df[col_name])
		df.drop(columns = [f'{col_name}_new'], inplace = True)

	### Step 3: Add UniProt metadata ###

	# Pre-split the multi-value columns in df_B into lists
	multi_cols = ['GO_ids', 'EC_number', 'cath_ids']
	for col_name in multi_cols:
		uniprot_df[col_name] = uniprot_df[col_name].fillna('').astype(str).str.split('_')

	# Explode chain_set into one row per (basename, pdb_id, chain_id)
	exp = (df[['basename', 'pdb_id', 'chain_set']].assign(chain_id=lambda d: d['chain_set'].str.split('-')).explode('chain_id'))

	# Join annotations from uniprot_df on (pdb_id, chain_id)
	merged = exp.merge(uniprot_df, on=['pdb_id', 'chain_id'], how='left')

	# Collapse to one row per basename, collecting unique values as sets
	annot_cols = ['uniprot_id', 'taxon_id', 'species_name', 'entry_name', 'GO_ids', 'protein_name', 'EC_number', 'cath_ids']
	agg = merged.groupby('basename')[annot_cols].agg(to_set)

	# Attach back to main df
	df = df.merge(agg, on = 'basename', how = 'left')
	df.dropna(subset = ['SMILES'], inplace = True)

	### Step 4: final cleaning ###
	df['n_uniprot'] = df['uniprot_id'].apply(len)
	set_cols = ['uniprot_id', 'taxon_id', 'species_name', 'entry_name', 'GO_ids', 'protein_name', 'EC_number', 'cath_ids']
	for col_name in set_cols:
		df[col_name] = df[col_name].apply(lambda x: sorted(x) if isinstance(x, set) else x)

	subset = df[df['n_uniprot'] > 0].copy()
	subset['uniprot_single'] = subset['uniprot_id'].apply(lambda x: x[0])

	### Step 5: add RMSD data ###
	subset = subset.merge(rmsd_df, how = 'inner', on = ['basename'])
	subset.dropna(subset = ['Ligand_RMSD', 'Pocket_RMSD', 'Scaffold_RMSD'], inplace = True)
	subset.to_parquet(f'{DATA_DIR}/metadata/CROWN_metadata.parquet', index = False)

	print(f'Original length: {len(rmsd_df)} - Final length: {len(subset)}')
