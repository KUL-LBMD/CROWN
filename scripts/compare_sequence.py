from src.config import DATA_DIR

import subprocess
import json
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from Bio.Align import PairwiseAligner, substitution_matrices
from tqdm import tqdm


class SequenceComparer:
    """
    Per CROWN complex-pair similarity on two levels:

      - protein-level: optimal one-to-one chain assignment (Hungarian on the mmseqs2
        fraction-identity matrix), scored as the chain-length-weighted mean fident.
      - pocket-level:  Smith-Waterman alignment of each mapped chain pair, then three
        variants (all normalized by |B_a|, following PLINDER Appendix B.2):
          pocket_shared          - A pocket residues aligned to a B pocket residue
          pocket_identity        - A pocket residues aligned to an identical residue
          pocket_identity_shared - aligned to an identical B pocket residue (both)

    Both metrics are anchored on complex A = the complex with fewer chains (ties broken
    lexicographically), so a pair (X, Y) gives the same result regardless of order.
    """

    # BLOSUM62 alphabet; any other character is treated as unknown ('X') for alignment.
    _VALID_AA = set("ARNDCQEGHILKMFPSTWYVBZX")

    def __init__(self, pocket_protein_gate=0.0):
        # If protein-level similarity for a pair is below this, skip the (expensive)
        # Smith-Waterman step and set pocket similarity to 0.0. 0.0 = compute everything
        # (exact). Raising it to ~0.2-0.3 is a big speed-up on diverse sets, at the cost
        # of missing the rare convergent-pocket case (low global identity, similar site).
        self.pocket_protein_gate = pocket_protein_gate

    # ------------------------------------------------------------------ mmseqs2

    def _run_mmseqs2(self, fasta_path, result_path, tmp_dir='tmp'):
        cmds = [
            f"mmseqs createdb {fasta_path} queryDB",
            f"mmseqs search queryDB queryDB resultDB {tmp_dir}",
            f"mmseqs convertalis queryDB queryDB resultDB {result_path} --format-output 'query,target,fident'",
        ]
        for cmd in cmds:
            subprocess.run(cmd, shell=True, check=True)

    def _build_similarity_matrix(self, result_path, id_list):
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

        # Vectorized assignment (symmetrized)
        mat[df["i"].values, df["j"].values] = df["fident"].values
        mat[df["j"].values, df["i"].values] = df["fident"].values

        return mat

    # ------------------------------------------------------------- sequence I/O

    @staticmethod
    def _load_sequences(fasta_path):
        """Parse a (multi-record) FASTA into {chain_id: sequence}. Chain IDs are the
        header tokens up to the first whitespace, i.e. '{complex}_{chain}'."""
        sequences = {}
        header, chunks = None, []
        with open(fasta_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        sequences[header] = "".join(chunks)
                    header = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line)
        if header is not None:
            sequences[header] = "".join(chunks)
        return sequences

    @staticmethod
    def _group_chains_by_complex(chain_ids, complex_ids):
        """Group '{complex}_{chain}' IDs under their complex using longest-prefix
        matching against the known complex-ID set (robust to complex IDs that
        themselves contain underscores)."""
        complex_ids_sorted = sorted(complex_ids, key=len, reverse=True)
        complex_to_chains = defaultdict(list)
        unassigned = []
        for cid in chain_ids:
            match = None
            for comp in complex_ids_sorted:
                prefix = comp + "_"
                if cid.startswith(prefix) and len(cid) > len(prefix):
                    match = comp
                    break
            if match is None:
                unassigned.append(cid)
            else:
                complex_to_chains[match].append(cid)
        return complex_to_chains, unassigned

    # --------------------------------------------------------------- alignment

    @staticmethod
    def _make_aligner():
        aligner = PairwiseAligner()
        aligner.mode = "local"  # Smith-Waterman
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -11
        aligner.extend_gap_score = -1
        return aligner

    def _sanitize(self, seq):
        """Coerce any non-BLOSUM62 character to 'X'. Length-preserving, so residue
        indices (and therefore binding-site seq_index values) stay valid."""
        return "".join(c if c in self._VALID_AA else "X" for c in seq)

    def _index_map(self, seq_a, seq_b, aligner):
        """Return {index_in_a: index_in_b} for every column of the best local
        alignment where both sequences have a residue (i.e. ungapped columns)."""
        if not seq_a or not seq_b:
            return {}
        try:
            alignment = aligner.align(seq_a, seq_b)[0]
        except (ValueError, IndexError):
            return {}
        a_blocks, b_blocks = alignment.aligned  # matched, equal-length blocks
        idx_map = {}
        for (a0, a1), (b0, b1) in zip(a_blocks, b_blocks):
            for k in range(a1 - a0):
                idx_map[a0 + k] = b0 + k
        return idx_map

    # ------------------------------------------------------------- pair scoring

    def _compare_pair(self, id1, id2, complex_to_chains, sequences,
                      chain_idx, sim_matrix, binding_by_complex, aligner):
        chains1 = complex_to_chains.get(id1, [])
        chains2 = complex_to_chains.get(id2, [])
        if not chains1 or not chains2:
            return np.nan, np.nan, np.nan, np.nan

        # A = fewer chains (deterministic tie-break -> guarantees order-independence)
        if len(chains1) < len(chains2) or (len(chains1) == len(chains2) and id1 <= id2):
            a_id, b_id, chains_a, chains_b = id1, id2, chains1, chains2
        else:
            a_id, b_id, chains_a, chains_b = id2, id1, chains2, chains1

        # fident submatrix (A rows x B cols) and optimal one-to-one assignment.
        a_rows = [chain_idx[c] for c in chains_a]
        b_cols = [chain_idx[c] for c in chains_b]
        sub = sim_matrix[np.ix_(a_rows, b_cols)]              # (nA, nB), nA <= nB
        row_ind, col_ind = linear_sum_assignment(-sub)        # maximize total fident
        mapping = {chains_a[r]: chains_b[c] for r, c in zip(row_ind, col_ind)}

        # Protein-level: fident of mapped pairs, weighted by A chain lengths.
        weights = np.array([len(sequences[chains_a[r]]) for r in row_ind], dtype=float)
        fidents = np.array([sub[r, c] for r, c in zip(row_ind, col_ind)], dtype=float)
        protein_score = float(np.average(fidents, weights=weights)) if weights.sum() else np.nan

        if self.pocket_protein_gate > 0.0 and protein_score < self.pocket_protein_gate:
            return protein_score, 0.0, 0.0, 0.0

        # Pocket-level metrics, all normalized by |B_a| (PLINDER Appendix B.2). For each
        # A pocket residue, project it through the SW alignment to residue q in B:
        #   pocket_shared          : q exists and is itself a B pocket residue
        #   pocket_identity        : q exists and is the same amino acid
        #   pocket_identity_shared : both of the above
        binding_a = binding_by_complex.get(a_id, {})
        binding_b = binding_by_complex.get(b_id, {})
        n_pocket = shared = identical = identical_shared = 0
        for chain_a, positions in binding_a.items():
            n_pocket += len(positions)  # every A pocket residue is in the denominator
            chain_b = mapping.get(chain_a)
            if chain_b is None or chain_a not in sequences or chain_b not in sequences:
                continue
            b_site = set(binding_b.get(chain_b, []))
            seq_a, seq_b = sequences[chain_a], sequences[chain_b]
            idx_map = self._index_map(self._sanitize(seq_a), self._sanitize(seq_b), aligner)
            for pos in positions:
                q = idx_map.get(pos)
                if q is None:
                    continue  # aligns to a gap -> shared/identical both fail
                is_shared = q in b_site
                is_identical = seq_a[pos] == seq_b[q] and seq_a[pos] != "X"
                shared += is_shared
                identical += is_identical
                identical_shared += is_shared and is_identical

        if n_pocket:
            pocket_shared = shared / n_pocket
            pocket_identity = identical / n_pocket
            pocket_identity_shared = identical_shared / n_pocket
        else:
            pocket_shared = pocket_identity = pocket_identity_shared = np.nan

        return protein_score, pocket_shared, pocket_identity, pocket_identity_shared

    # ---------------------------------------------------------------- pipeline

    def main(self):
        fasta_path = f'{DATA_DIR}/crown_sequences.fasta'
        result_path = f'{DATA_DIR}/crown_seqsim.m8'

        self._run_mmseqs2(fasta_path, result_path)

        # Canonical chain list from the FASTA (includes chains with no non-self hits).
        sequences = self._load_sequences(fasta_path)
        chain_ids = sorted(sequences.keys())
        chain_idx = {cid: i for i, cid in enumerate(chain_ids)}

        sim_matrix = self._build_similarity_matrix(result_path, chain_ids)
        pd.DataFrame(sim_matrix, index=chain_ids, columns=chain_ids).to_hdf(
            f'{DATA_DIR}/crown_chain_seqsim.h5', key='sim', complevel=5, complib='blosc')

        with open(f'{DATA_DIR}/binding_residues.json') as f:
            binding_residues = json.load(f)

        complex_ids = sorted(binding_residues.keys())
        complex_to_chains, unassigned = self._group_chains_by_complex(chain_ids, complex_ids)
        if unassigned:
            print(f"[warn] {len(unassigned)} chains not matched to a complex, "
                  f"e.g. {unassigned[:5]}")

        # Binding-site indices grouped by full chain_id ('{complex}_{chain}').
        binding_by_complex = {}
        for comp, records in binding_residues.items():
            by_chain = defaultdict(list)
            for rec in records:
                by_chain[f"{comp}_{rec['chain']}"].append(rec['seq_index'])
            binding_by_complex[comp] = by_chain

        aligner = self._make_aligner()

        n = len(complex_ids)
        protein_sim = np.eye(n, dtype=np.float32)
        pk_shared = np.eye(n, dtype=np.float32)
        pk_identity = np.eye(n, dtype=np.float32)
        pk_identity_shared = np.eye(n, dtype=np.float32)
        records = []  # tidy long-form rows, one per unordered pair (i < j)

        total_pairs = n * (n - 1) // 2
        for i, j in tqdm(combinations(range(n), 2), total=total_pairs,
                         desc="Comparing complexes", unit="pair"):
            prot, sh, ident, ident_sh = self._compare_pair(
                complex_ids[i], complex_ids[j], complex_to_chains, sequences,
                chain_idx, sim_matrix, binding_by_complex, aligner)
            protein_sim[i, j] = protein_sim[j, i] = prot
            pk_shared[i, j] = pk_shared[j, i] = sh
            pk_identity[i, j] = pk_identity[j, i] = ident
            pk_identity_shared[i, j] = pk_identity_shared[j, i] = ident_sh
            records.append((complex_ids[i], complex_ids[j], prot, sh, ident, ident_sh))

        for mat, name in [(protein_sim, 'protein_sim'),
                          (pk_shared, 'pocket_shared'),
                          (pk_identity, 'pocket_identity'),
                          (pk_identity_shared, 'pocket_identity_shared')]:
            pd.DataFrame(mat, index=complex_ids, columns=complex_ids).to_hdf(
                f'{DATA_DIR}/crown_{name}.h5', key='sim', complevel=5, complib='blosc')

        # Tidy long-form table (one row per unordered pair) for threshold-based
        # filtering / dedup. Requires a parquet engine (pyarrow or fastparquet).
        pd.DataFrame(records, columns=['id1', 'id2', 'protein_sim', 'pocket_shared',
                                       'pocket_identity', 'pocket_identity_shared']).to_parquet(
            f'{DATA_DIR}/crown_pair_similarity.parquet', index=False)


if __name__ == '__main__':
    SequenceComparer().main()