import subprocess
import pandas as pd
import numpy as np
import requests
import os
import shutil

from src.config import DATA_DIR

def fetch_sequences(uniprot_ids, fasta_path):
    with open(fasta_path, 'w') as f:
        batch_size = 100
        for i in range(0, len(uniprot_ids), batch_size):

            if i % 1000 == 0:
                print(f"Fetching {i}/{len(uniprot_ids)}...")

            batch = uniprot_ids[i:i+batch_size]
            query = "+OR+".join(f"accession:{uid}" for uid in batch)
            url = f"https://rest.uniprot.org/uniprotkb/stream?query={query}&format=fasta"
            resp = requests.get(url)
            f.write(resp.text)

def simplify_fasta_headers(fasta_path):
    from Bio import SeqIO
    records = list(SeqIO.parse(fasta_path, "fasta"))
    for rec in records:
        # extract accession from "sp|P12345|NAME" or "tr|Q67890|NAME"
        parts = rec.id.split("|")
        rec.id = parts[1] if len(parts) >= 2 else rec.id
        rec.description = ""
    SeqIO.write(records, fasta_path, "fasta")

# Step 2: Run MMseqs2 all-vs-all
def run_mmseqs2(fasta_path, tmp_dir="tmp", result_path="results.m8"):
    cmds = [
        f"mmseqs createdb {fasta_path} queryDB",
        f"mmseqs search queryDB queryDB resultDB {tmp_dir}",
        f"mmseqs convertalis queryDB queryDB resultDB {result_path} --format-output 'query,target,fident'",
    ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True, check=True)

# Step 3: Parse into a similarity matrix
def build_similarity_matrix(result_path, id_list):
    id_to_idx = {uid: i for i, uid in enumerate(id_list)}
    n = len(id_list)
    mat = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(mat, 1.0)

    df = pd.read_csv(result_path, sep="\t", header=None,
                     names=["query", "target", "fident"])

    # Map to indices, drop pairs not in our list
    df["i"] = df["query"].map(id_to_idx)
    df["j"] = df["target"].map(id_to_idx)
    df = df.dropna(subset=["i", "j"]).astype({"i": int, "j": int})

    # Vectorized assignment
    mat[df["i"].values, df["j"].values] = df["fident"].values
    mat[df["j"].values, df["i"].values] = df["fident"].values

    return mat

def cleanup_mmseqs(fasta_path, result_path, tmp_dir="tmp",
                   db_prefixes=("queryDB", "resultDB")):
    for prefix in db_prefixes:
        subprocess.run(f"mmseqs rmdb {prefix}", shell=True, check=False)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for path in (fasta_path, result_path, result_path + ".dbtype"):
        if os.path.exists(path):
            os.remove(path)

def cluster_mmseqs():
    df = pd.read_parquet(f'{DATA_DIR}/metadata/CROWN_metadata.parquet')
    uniprot_ids = df['uniprot_single'].unique().tolist()

    fasta_path = 'sequences.fasta'
    result_path = 'results.m8'
    fetch_sequences(uniprot_ids, fasta_path)
    simplify_fasta_headers(fasta_path)
    run_mmseqs2(fasta_path, result_path = result_path)
    
    sim_matrix = build_similarity_matrix(result_path, uniprot_ids)
    sim_df = pd.DataFrame(sim_matrix, index = uniprot_ids, columns = uniprot_ids)
    sim_df.to_hdf(f'{DATA_DIR}/metadata/CROWN_seqsim.h5', key = 'sim', complevel = 5, complib = 'blosc')
    cleanup_mmseqs(fasta_path, result_path)
