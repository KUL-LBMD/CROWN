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
	posebusters_df = pd.read_csv(f'{DATA_DIR}/metadata/posebusters.csv')

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
	new_species_names = []
	for row in df.itertuples():
		new_name = row.source_organism_ncbi
		old_names = set(row.species_name)
		old_names.add(new_name)
		new_species_names.append(list(old_names))
	df['species_name'] = new_species_names
	df.drop(columns = ['source_organism', 'source_organism_ncbi'], inplace = True)

	set_cols = ['uniprot_id', 'taxon_id', 'species_name', 'entry_name', 'GO_ids', 'protein_name', 'EC_number', 'cath_ids']
	for col_name in set_cols:
		df[col_name] = df[col_name].apply(lambda x: sorted(x) if isinstance(x, set) else x)

	for col in ['uniprot_id', 'taxon_id', 'species_name', 'entry_name', 'GO_ids', 'protein_name', 'EC_number', 'cath_ids']:
		df[col] = df[col].apply(lambda x: ';'.join([str(y) for y in x]))

	df.rename(columns = {'uniprot_id': 'uniprot_ids', 'taxon_id': 'taxon_ids', 'species_name': 'species_names',
		'entry_name': 'entry_names', 'protein_name': 'protein_names', 'EC_number': 'EC_numbers'},
		inplace = True)

	### Step 5: add RMSD data ###
	df = df.merge(rmsd_df, how = 'inner', on = ['basename'])
	df.dropna(subset = ['Ligand_RMSD', 'Pocket_RMSD', 'Scaffold_RMSD'], inplace = True)
	print(f'Original length: {len(rmsd_df)} - Final length: {len(df)}')

	### Step 6: add PoseBusters data ###
	subset = posebusters_df[posebusters_df['entry_id'].isin(df['basename'])]
	bool_cols = ['minimum_distance_to_protein', 'minimum_distance_to_waters', 'minimum_distance_to_organic_cofactors', 'internal_steric_clash',
		'double_bond_flatness', 'non-aromatic_ring_non-flatness', 'bond_lengths', 'internal_energy', 'pb_valid']

	for col in bool_cols:
		subset[col] = subset[col].astype(bool)
	subset[bool_cols] = ~subset[bool_cols]

	subset = subset[['entry_id', 'state', 'minimum_distance_to_protein', 'minimum_distance_to_waters', 'minimum_distance_to_organic_cofactors',
                 'internal_steric_clash', 'double_bond_flatness', 'non-aromatic_ring_non-flatness', 'bond_lengths', 'internal_energy', 'pb_valid']]

	subset1 = subset[subset['state'] == 'raw']
	subset2 = subset[subset['state'] == 'minimized']

	subset1.rename(columns = {'entry_id': 'basename', 'minimum_distance_to_protein': 'PB protein steric clash (raw)', 'minimum_distance_to_waters': 'PB water steric clash (raw)',
                          'minimum_distance_to_organic_cofactors': 'PB cofactor steric clash (raw)', 'internal_steric_clash': 'PB internal steric clash (raw)',
                          'double_bond_flatness': 'PB non-flat double bonds (raw)', 'non-aromatic_ring_non-flatness': 'PB flat non-aromatic ring (raw)',
                          'bond_lengths': 'PB invalid bond length (raw)', 'pb_valid': 'PB any failure (raw)'}, inplace = True)

	subset2.rename(columns = {'entry_id': 'basename', 'minimum_distance_to_protein': 'PB protein steric clash (min)', 'minimum_distance_to_waters': 'PB water steric clash (min)',
                          'minimum_distance_to_organic_cofactors': 'PB cofactor steric clash (min)', 'internal_steric_clash': 'PB internal steric clash (min)',
                          'double_bond_flatness': 'PB non-flat double bonds (min)', 'non-aromatic_ring_non-flatness': 'PB flat non-aromatic ring (min)',
                          'bond_lengths': 'PB invalid bond length (min)', 'pb_valid': 'PB any failure (min)'}, inplace = True)

	subset1.drop(columns = ['state', 'internal_energy'], inplace = True)
	subset2.drop(columns = ['state', 'internal_energy'], inplace = True)

	df = pd.merge(df, subset1, how = 'left', on = ['basename'])
	df = pd.merge(df, subset2, how = 'left', on = ['basename'])

	df.to_parquet('CROWN_metadata.parquet', index = False)
	df.to_csv('CROWN_metadata.csv', index = False)
