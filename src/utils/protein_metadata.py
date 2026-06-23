import pandas as pd
import os
import sys
import re
import gzip
import requests
import urllib.request
from collections import defaultdict

from src.config import DATA_DIR

def download_swissprot():

    print('Starting SwissProt download')
    out_path = f'{DATA_DIR}/metadata/uniprot_swissprot.tsv.gz'

    params = {
        "query": "reviewed:true",
        "format": "tsv",
        "fields": ("accession,id,organism_name,organism_id,"
                   "go_id,protein_name,ec"),
        "compressed": "true",          # server returns gzip bytes
    }
    headers = {
        # Be a good API citizen — and some proxies/servers drop the default
        # python-requests UA. Put a real contact here.
        "User-Agent": "yourname-swissprot-dl/1.0 (you@uni.edu)",
    }
    url = "https://rest.uniprot.org/uniprotkb/stream"

    with requests.get(url, params=params, headers=headers,
                      stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:        # already gzip — write raw bytes
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

def download_cath():

    print('Starting CATH download')
    url = "http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"
    urllib.request.urlretrieve(url, f'{DATA_DIR}/metadata/cath-domain-list.txt')

    results = defaultdict(set)

    with open(f'{DATA_DIR}/metadata/cath-domain-list.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                parts = line.split()
                if len(parts) > 5:
                    pdb_id = parts[0][:4]
                    chain_id = parts[0][4]
                    cath_id = '.'.join(parts[1:5])

                    results[(pdb_id, chain_id)].add(cath_id)

    with open(f'{DATA_DIR}/metadata/cath_processed.csv', 'w') as f:

        f.write('pdb_id,chain_id,cath_ids\n')

        for key, value in results.items():
            pdb_id = key[0]
            chain_id = key[1]
            cath_ids = '_'.join(value)

            f.write(f'{pdb_id},{chain_id},{cath_ids}\n')

def download_sifts():
	print('Starting SIFTS download')
	url = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz"
	out_path = f'{DATA_DIR}/metadata/pdb_chain_uniprot.csv.gz'
	urllib.request.urlretrieve(url, out_path)

def get_protein_metadata():
	download_swissprot()
	download_cath()
	download_sifts()

	uniprot_df = pd.read_csv(f'{DATA_DIR}/metadata/uniprot_swissprot.tsv.gz', compression = 'gzip', sep = '\t')
	cath_df = pd.read_csv(f'{DATA_DIR}/metadata/cath_processed.csv')
	sifts_df = pd.read_csv(f'{DATA_DIR}/metadata/pdb_chain_uniprot.csv.gz', compression = 'gzip', skiprows = 1)

	uniprot_df.rename(columns = {'Entry': 'uniprot_id', 'Entry Name': 'entry_name', 'Organism': 'species_name',
		'Organism (ID)': 'taxon_id', 'Gene Ontology IDs': 'GO_ids', 'Protein names': 'protein_name', 'EC number': 'EC_number'},
		inplace = True)
	uniprot_df.drop_duplicates(subset = ['uniprot_id'], keep = 'first', inplace = True)

	sifts_df.rename(columns = {'PDB': 'pdb_id', 'CHAIN': 'chain_id', 'SP_PRIMARY': 'uniprot_id'}, inplace = True)
	sifts_df.drop_duplicates(subset = ['pdb_id', 'chain_id'], keep = 'first', inplace = True)

	uniprot_df['GO_ids'] = uniprot_df['GO_ids'].str.replace('; ', '_')
	uniprot_df['EC_number'] = uniprot_df['EC_number'].str.replace('; ', '_')

	sifts_df = sifts_df.merge(uniprot_df, how = 'left', on = ['uniprot_id'])
	sifts_df = sifts_df.merge(cath_df, how = 'left', on = ['pdb_id', 'chain_id'])
	sifts_df.to_csv(f'{DATA_DIR}/metadata/uniprot_metadata.csv', index = False)
