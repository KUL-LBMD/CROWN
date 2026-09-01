from src.config import DATA_DIR

import os
import shutil
import subprocess
import json
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
import pandas as pd
import gemmi
from scipy.spatial import KDTree
from biopandas.mol2 import PandasMol2
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
from Bio.Align import PairwiseAligner, substitution_matrices
from tqdm import tqdm

def cleanup_mmseqs(tmp_dir="tmp", db_prefixes=("queryDB", "resultDB")):
    for prefix in db_prefixes:
        subprocess.run(f"mmseqs rmdb {prefix}", shell=True, check=False)
    shutil.rmtree(tmp_dir, ignore_errors=True)

_SHARED = {}  # populated in the parent, inherited by workers via fork (copy-on-write)


def _compare_worker(pair):
    """Score one candidate complex pair (given as complex indices). Reads the shared
    read-only inputs inherited from the parent process; returns the tidy edge row."""
    i, j = pair
    ids = _SHARED['complex_ids']
    id1, id2 = ids[i], ids[j]
    prot, sh, ident, ident_sh = _SHARED['comparer']._compare_pair(
        id1, id2, _SHARED['complex_to_chains'], _SHARED['sequences'],
        _SHARED['sim'], _SHARED['binding_by_complex'], _SHARED['aligner'])
    return id1, id2, prot, sh, ident, ident_sh

class FastaBuilder:
    """
    1. Create separate FASTA per protein chain in CROWN
    2. Make list of binding site residues per target (easily mappable to fasta files)
    """

    CONTACT_CUTOFF = 4.0  # Angstrom, heavy-atom to heavy-atom

    @staticmethod
    def _one_letter_code(resname):
        """Map a three-letter residue name to a one-letter code ('X' if unknown)."""
        info = gemmi.find_tabulated_residue(resname)
        if info is not None and info.one_letter_code.strip():
            return info.one_letter_code.upper()
        return 'X'

    @staticmethod
    def _load_ligand_heavy_atoms(ligand_path):
        """Return an (N, 3) array of ligand heavy-atom coordinates from a mol2 file."""
        pmol = PandasMol2().read_mol2(ligand_path)
        df = pmol.df
        # SYBYL atom types look like 'C.3', 'N.ar', 'O.2', 'H', 'Cl' ...
        # the element is everything before the first '.'
        elements = df['atom_type'].str.split('.', n=1).str[0]
        heavy = df[elements != 'H']
        return heavy[['x', 'y', 'z']].to_numpy()

    def _load_receptor(self, receptor_path):
        """
        Parse the receptor and return:
          - coords:        (M, 3) array of heavy-atom coordinates
          - atom_res_ref:  length-M list of (chain_name, seq_index), one per atom
          - sequences:     {chain_name: one_letter_sequence}
          - residue_meta:  {(chain_name, seq_index): {res_seqid, icode, res_name, one_letter}}

        seq_index is the 0-based position of the residue within that chain's FASTA
        sequence, so a binding residue maps back to the FASTA as sequences[chain][seq_index].
        The sequence and the per-atom references are built in the same pass, which
        guarantees the indices stay aligned.
        """
        st = gemmi.read_structure(receptor_path)
        st.setup_entities()  # ensure polymers are detected for plain PDB input
        model = st[0]

        coords = []
        atom_res_ref = []
        sequences = {}
        residue_meta = {}

        for chain in model:
            polymer = chain.get_polymer()  # protein residues only (skips waters/ligands/ions)
            if len(polymer) == 0:
                continue

            seq_codes = []
            for seq_index, res in enumerate(polymer):
                code = self._one_letter_code(res.name)
                seq_codes.append(code)
                residue_meta[(chain.name, seq_index)] = {
                    'res_seqid': res.seqid.num,
                    'icode': res.seqid.icode.strip(),
                    'res_name': res.name,
                    'one_letter': code,
                }
                for atom in res:
                    if atom.is_hydrogen():
                        continue
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    atom_res_ref.append((chain.name, seq_index))

            sequences[chain.name] = ''.join(seq_codes)

        coords = np.asarray(coords, dtype=float)
        return coords, atom_res_ref, sequences, residue_meta

    def _write_fasta(self, subdir, sequences, line_width=60):
        """Write one FASTA file per chain to {DATA_DIR}/metadata/crown_sequences.fasta"""

        path = f'{DATA_DIR}/metadata/crown_sequences.fasta'
        with open(path, 'a') as fh:
            for chain_name, seq in sequences.items():
                fh.write(f'>{subdir}_{chain_name}\n')
                for i in range(0, len(seq), line_width):
                    fh.write(seq[i:i + line_width] + '\n')

    def _process_complex(self, subdir, receptor_path, ligand_path):
        """
        - Store FASTA files for all receptor chains in f'{DATA_DIR}/fasta_files'
        - Binding site residues defined as all residues with any heavy-atom contact
          with the ligand within 4Å

        Returns a list of binding-site residue records, each easily mappable back to
        the chain FASTA via (chain, seq_index).
        """
        coords, atom_res_ref, sequences, residue_meta = self._load_receptor(receptor_path)
        self._write_fasta(subdir, sequences)

        lig_coords = self._load_ligand_heavy_atoms(ligand_path)

        if coords.size == 0 or lig_coords.size == 0:
            return []

        # For every ligand heavy atom, collect receptor heavy atoms within the cutoff.
        # Tree is built on the (larger) receptor set and queried with the ligand atoms.
        tree = KDTree(coords)
        neighbor_lists = tree.query_ball_point(lig_coords, r=self.CONTACT_CUTOFF)

        contact_atom_idx = set()
        for nl in neighbor_lists:
            contact_atom_idx.update(nl)

        contact_residues = {atom_res_ref[i] for i in contact_atom_idx}

        binding_residues = []
        for key in sorted(contact_residues):  # sorted by (chain, seq_index)
            chain_name, seq_index = key
            binding_residues.append(
                {'chain': chain_name, 'seq_index': seq_index, **residue_meta[key]}
            )

        return binding_residues

    def main(self):
        binding_residues_dict = defaultdict(list)

        subdirs = sorted(os.listdir(f'{DATA_DIR}/complexes'))
        for subdir in tqdm(subdirs, desc='Processing complexes', unit='complex'):
            receptor_path = f'{DATA_DIR}/complexes/{subdir}/receptor_minimized.pdb'
            ligand_path = f'{DATA_DIR}/mol2_files/{subdir}/ligand_minimized.mol2'

            new_binding_residues = self._process_complex(subdir, receptor_path, ligand_path)
            binding_residues_dict[subdir] = new_binding_residues

        # Store as JSON: keyed by complex, each value a list of residue records that
        # map straight back to the per-chain FASTA files via (chain, seq_index).
        out_path = f'{DATA_DIR}/metadata/binding_residues.json'
        with open(out_path, 'w') as fh:
            json.dump(binding_residues_dict, fh, indent=2)


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

    Memory note
    -----------
    The chain-vs-chain fident matrix is kept SPARSE throughout. mmseqs2 only reports
    chain pairs with a hit, so an n_chains x n_chains dense array (n^2 * 4 bytes, plus a
    second copy when wrapped in a DataFrame for HDF5) is almost entirely zeros and is the
    thing that gets OOM-killed. Instead we hold a nested dict {chain: {chain: fident}}
    whose size scales with the number of hits, and build the tiny per-complex-pair blocks
    on demand. The saved artifact is likewise a scipy CSR matrix (.npz) + an ids file
    rather than a dense HDF5 table.
    """

    # BLOSUM62 alphabet; any other character is treated as unknown ('X') for alignment.
    _VALID_AA = set("ARNDCQEGHILKMFPSTWYVBZX")

    def __init__(self, pocket_protein_gate=0.0, edge_floor=0.3,
                 cluster_metric='protein_sim', cluster_threshold=0.5,
                 compute_pocket=True):
        # If protein-level similarity for a pair is below this, skip the (expensive)
        # Smith-Waterman step and set pocket similarity to 0.0. 0.0 = compute everything
        # (exact). Raising it to ~0.2-0.3 is a big speed-up on diverse sets, at the cost
        # of missing the rare convergent-pocket case (low global identity, similar site).
        self.pocket_protein_gate = pocket_protein_gate

        # If you only cluster on protein_sim, set this False to skip Smith-Waterman
        # entirely (pocket metrics come back as NaN). SW is the dominant per-pair cost, so
        # this is the single biggest speed-up when the pocket columns aren't needed.
        self.compute_pocket = compute_pocket

        # Persist only complex pairs whose best metric clears this floor. Set it to the
        # LOWEST similarity threshold you might ever cluster at: a pair below the floor on
        # every metric can never be an edge at any threshold >= floor, so dropping it
        # cannot change a single cluster label. This is what lets us skip the dense
        # n_complexes^2 matrices AND the full O(n^2) pair list. Lower floor = safer but
        # bigger edge set; higher = leaner. 0.0 keeps everything (defeats the purpose).
        self.edge_floor = edge_floor

        # Metric + threshold used to write crown_cluster_labels.parquet in main(). You can
        # re-cluster later at any metric and any threshold >= edge_floor straight from the
        # saved edge parquet (see _single_linkage_labels) without recomputing anything.
        self.cluster_metric = cluster_metric
        self.cluster_threshold = cluster_threshold

        # Metric + threshold used to write crown_cluster_labels.parquet in main(). You can
        # re-cluster later at any metric and any threshold >= edge_floor straight from the
        # saved edge parquet (see _single_linkage_labels) without recomputing anything.
        self.cluster_metric = cluster_metric
        self.cluster_threshold = cluster_threshold

    # ------------------------------------------------------------------ mmseqs2

    def _run_mmseqs2(self, fasta_path, result_path, tmp_dir='tmp'):
        cmds = [
            f"mmseqs createdb {fasta_path} queryDB",
            f"mmseqs search queryDB queryDB resultDB {tmp_dir} "
            f"-e 0.01 --min-seq-id 0.2 --max-seqs 5000",
            f"mmseqs convertalis queryDB queryDB resultDB {result_path} "
            f"--format-output 'query,target,fident'",
        ]
        for cmd in cmds:
            subprocess.run(cmd, shell=True, check=True)

    def _build_similarity_lookup(self, result_path, chunksize=5_000_000):
        """Stream the mmseqs2 result table into a sparse, symmetric lookup:
        {chain: {chain: fident}}. Diagonal (self-identity = 1.0) is implicit and
        handled at lookup/save time, so we never allocate an n_chains^2 array.

        Peak memory is O(number of reported hits), not O(n_chains^2). The file is read
        in chunks so even a very large .m8 never lands in memory all at once.
        """
        sim = defaultdict(dict)
        reader = pd.read_csv(
            result_path, sep="\t", header=None,
            names=["query", "target", "fident"], usecols=[0, 1, 2],
            dtype={"query": str, "target": str, "fident": np.float32},
            chunksize=chunksize,
        )
        for chunk in reader:
            q = chunk["query"].to_numpy()
            t = chunk["target"].to_numpy()
            f = chunk["fident"].to_numpy()
            for qi, ti, fi in zip(q, t, f):
                if qi == ti:
                    continue  # self-hit; diagonal is implicit
                fi = float(fi)
                sim[qi][ti] = fi
                sim[ti][qi] = fi  # symmetrize
        return sim

    def _save_chain_similarity(self, sim, chain_ids, npz_path, ids_path):
        """Persist the chain-vs-chain fident matrix as a sparse CSR matrix (.npz) plus a
        JSON list giving the row/column order. Reconstruct downstream with:

            import scipy.sparse as sp, json
            mat = sp.load_npz(npz_path)                       # CSR, n_chains x n_chains
            ids = json.load(open(ids_path))                   # chain order
            # optional dense DataFrame (only if it fits!):
            # import pandas as pd
            # df = pd.DataFrame(mat.toarray(), index=ids, columns=ids)
        """
        idx = {cid: i for i, cid in enumerate(chain_ids)}
        rows, cols, vals = [], [], []
        for q, targets in sim.items():
            qi = idx.get(q)
            if qi is None:
                continue
            for t, fi in targets.items():
                ti = idx.get(t)
                if ti is None:
                    continue
                rows.append(qi)
                cols.append(ti)
                vals.append(fi)

        n = len(chain_ids)
        diag = np.arange(n, dtype=np.int32)
        rows = np.concatenate([np.asarray(rows, dtype=np.int32), diag])
        cols = np.concatenate([np.asarray(cols, dtype=np.int32), diag])
        vals = np.concatenate([np.asarray(vals, dtype=np.float32),
                               np.ones(n, dtype=np.float32)])

        mat = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        sp.save_npz(npz_path, mat)
        with open(ids_path, "w") as fh:
            json.dump(chain_ids, fh)

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

    def _submatrix(self, chains_a, chains_b, sim):
        """Build the small (nA x nB) fident block for two complexes directly from the
        sparse lookup. Missing pairs default to 0.0; identical chain IDs to 1.0."""
        nA, nB = len(chains_a), len(chains_b)
        sub = np.zeros((nA, nB), dtype=np.float32)
        for r, ca in enumerate(chains_a):
            row = sim.get(ca, {})
            for c, cb in enumerate(chains_b):
                sub[r, c] = 1.0 if ca == cb else row.get(cb, 0.0)
        return sub

    def _compare_pair(self, id1, id2, complex_to_chains, sequences,
                      sim, binding_by_complex, aligner):
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
        sub = self._submatrix(chains_a, chains_b, sim)        # (nA, nB), nA <= nB
        row_ind, col_ind = linear_sum_assignment(-sub)        # maximize total fident
        mapping = {chains_a[r]: chains_b[c] for r, c in zip(row_ind, col_ind)}

        # Protein-level: fident of mapped pairs, weighted by A chain lengths.
        weights = np.array([len(sequences[chains_a[r]]) for r in row_ind], dtype=float)
        fidents = np.array([sub[r, c] for r, c in zip(row_ind, col_ind)], dtype=float)
        protein_score = float(np.average(fidents, weights=weights)) if weights.sum() else np.nan

        if self.pocket_protein_gate > 0.0 and protein_score < self.pocket_protein_gate:
            return protein_score, 0.0, 0.0, 0.0

        if not self.compute_pocket:
            return protein_score, np.nan, np.nan, np.nan  # skip Smith-Waterman entirely

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

    # --------------------------------------------------------------- clustering

    @staticmethod
    def _single_linkage_labels(edges, complex_ids, metric, threshold):
        """Single-linkage cluster labels via connected components.

        Single-linkage clustering at a similarity threshold t is identical to the
        connected components of the graph whose edges are the complex pairs with
        `metric` >= t. Complexes with no qualifying edge fall out as singletons because
        every complex is present as a node. Returns an int label array aligned to
        `complex_ids` (so labels[k] is the cluster of complex_ids[k]).

        This touches only the above-threshold edges, so it stays cheap no matter how many
        complexes there are, and can be re-run at any threshold >= edge_floor without
        recomputing the pairwise scores.
        """
        idx = {c: i for i, c in enumerate(complex_ids)}
        n = len(complex_ids)
        sel = edges[edges[metric] >= threshold]
        i = sel['id1'].map(idx).to_numpy().astype(np.int32)
        j = sel['id2'].map(idx).to_numpy().astype(np.int32)
        data = np.ones(len(i), dtype=np.int8)
        graph = sp.coo_matrix((data, (i, j)), shape=(n, n))
        _, labels = connected_components(graph, directed=False, connection='weak')
        return labels

    @staticmethod
    def _build_mst(edges, complex_ids, metric):
        """Single-linkage merge tree as a minimum spanning forest for one metric.

        The single-linkage hierarchy is fully determined by the MST of the distance graph
        (d = 1 - similarity): cutting the dendrogram at height h and taking connected
        components of {edges with d <= h} are the same operation, and the MST preserves
        every such cut (it is the maximum-bottleneck spanning tree). Storing the forest
        (<= n_complexes - 1 edges) is therefore enough to relabel at ANY threshold, with
        no n^2 matrix ever formed.

        Returns a DataFrame [id1, id2, similarity] where `similarity` is the merge height
        (the metric value at which the endpoints' components join). Because `edges` only
        holds pairs with a metric >= edge_floor, the forest is valid for cuts at
        thresholds >= edge_floor; lower cuts would need the dropped edges.
        """
        idx = {c: i for i, c in enumerate(complex_ids)}
        n = len(complex_ids)
        sim = edges[metric].to_numpy(dtype=np.float64)
        keep = sim == sim  # drop NaN rows for this metric
        i = edges['id1'].map(idx).to_numpy()[keep].astype(np.int32)
        j = edges['id2'].map(idx).to_numpy()[keep].astype(np.int32)
        sim = sim[keep]

        # Minimise distance = 2 - sim (in [1, 2]): equivalent to maximising total
        # similarity, so the MST is the single-linkage merge tree. The +1 shift over the
        # usual 1 - sim keeps every weight strictly positive, so a perfect-similarity edge
        # (sim = 1) is not stored as a sparse zero and silently dropped. Adding a constant
        # to all edges leaves the MST unchanged (every spanning tree has n-1 edges).
        dist = 2.0 - sim
        graph = sp.coo_matrix((dist, (i, j)), shape=(n, n))
        mst = minimum_spanning_tree(graph).tocoo()
        return pd.DataFrame({
            'id1': [complex_ids[r] for r in mst.row],
            'id2': [complex_ids[c] for c in mst.col],
            'similarity': 2.0 - mst.data,  # recover merge-height similarity
        })

    @staticmethod
    def _labels_from_mst(mst, complex_ids, threshold):
        """Flat single-linkage labels at `threshold`, cut from a saved merge forest:
        connected components keeping only merges with similarity >= threshold. Nodes with
        no surviving edge are singletons. Valid for threshold >= edge_floor. Returns an
        int label array aligned to `complex_ids`. Cheap enough to call in a sweep."""
        idx = {c: i for i, c in enumerate(complex_ids)}
        n = len(complex_ids)
        sel = mst[mst['similarity'] >= threshold]
        i = sel['id1'].map(idx).to_numpy().astype(np.int32)
        j = sel['id2'].map(idx).to_numpy().astype(np.int32)
        data = np.ones(len(i), dtype=np.int8)
        graph = sp.coo_matrix((data, (i, j)), shape=(n, n))
        _, labels = connected_components(graph, directed=False, connection='weak')
        return labels

    # -------------------------------------------------------------- candidates

    @staticmethod
    def _candidate_pairs(sim, complex_to_chains, complex_ids):
        """Complex-index pairs (i < j) that share at least one chain-chain mmseqs hit.

        Any complex pair with no shared chain hit has an all-zero fident submatrix, so its
        protein_sim is 0 and it can never be an edge -- scoring it is wasted work. This
        turns the O(n_complexes^2) sweep into one bounded by the number of chain hits.

        Exactness: this is exact for protein-level clustering. For pocket-level it assumes
        pocket similarity implies detectable *sequence* similarity between the mapped
        chains (the standard mmseqs-prefilter assumption). If you care about remote
        homologs, raise mmseqs sensitivity (-s 7.5) and --max-seqs so no true hit is
        missed; a truncated search here would drop real candidate pairs.

        Returns (i_idx, j_idx) int64 arrays of unique upper-triangle pairs.
        """
        cidx = {c: k for k, c in enumerate(complex_ids)}
        chain_to_cidx = {}
        for comp, chs in complex_to_chains.items():
            k = cidx.get(comp)
            if k is not None:
                for ch in chs:
                    chain_to_cidx[ch] = k

        n = len(complex_ids)
        ii, jj = [], []
        for c1, targets in sim.items():
            a = chain_to_cidx.get(c1)
            if a is None:
                continue
            for c2 in targets:
                b = chain_to_cidx.get(c2)
                if b is None or b == a:
                    continue
                lo, hi = (a, b) if a < b else (b, a)
                ii.append(lo)
                jj.append(hi)

        if not ii:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        # Dedup via a single linear key (i * n + j) to keep memory to two int arrays.
        key = np.unique(np.asarray(ii, np.int64) * n + np.asarray(jj, np.int64))
        return key // n, key % n

    # ---------------------------------------------------------------- pipeline

    def main(self):
        fasta_path = f'{DATA_DIR}/metadata/crown_sequences.fasta'
        result_path = f'{DATA_DIR}/metadata/crown_seqsim.m8'

        self._run_mmseqs2(fasta_path, result_path)

        # Canonical chain list from the FASTA (includes chains with no non-self hits).
        sequences = self._load_sequences(fasta_path)
        chain_ids = sorted(sequences.keys())

        # Sparse chain-vs-chain fident lookup (streamed) instead of a dense n^2 array.
        sim = self._build_similarity_lookup(result_path)
        self._save_chain_similarity(
            sim, chain_ids,
            npz_path=f'{DATA_DIR}/metadata/crown_chain_seqsim.npz',
            ids_path=f'{DATA_DIR}/metadata/crown_chain_seqsim_ids.json')

        with open(f'{DATA_DIR}/metadata/binding_residues.json') as f:
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

        # Candidate generation: only complex pairs sharing a chain-chain mmseqs hit can be
        # similar, so the ~n^2/2 exhaustive comparisons collapse to those few. Everything
        # else is a guaranteed non-edge (all-zero fident -> protein_sim 0) and skipped.
        n = len(complex_ids)
        cand_i, cand_j = self._candidate_pairs(sim, complex_to_chains, complex_ids)
        print(f"[info] scoring {len(cand_i):,} candidate pairs "
              f"(exhaustive would be {n * (n - 1) // 2:,})")

        # Parallel scoring. Hungarian + Smith-Waterman per pair is embarrassingly parallel;
        # the big read-only inputs (sequences, sim, binding) are shared with workers via
        # fork (copy-on-write) rather than pickled per task. Requires a fork-capable OS
        # (Linux). Each worker builds its own aligner. Set compute_pocket=False to drop SW.
        _SHARED.update(comparer=self, complex_to_chains=complex_to_chains,
                       sequences=sequences, sim=sim, complex_ids=complex_ids,
                       binding_by_complex=binding_by_complex,
                       aligner=self._make_aligner())

        n_workers = 64
        records = []
        ctx = get_context('fork')
        with ctx.Pool(processes=n_workers) as pool:
            work = zip(cand_i.tolist(), cand_j.tolist())
            for id1, id2, prot, sh, ident, ident_sh in tqdm(
                    pool.imap_unordered(_compare_worker, work, chunksize=256),
                    total=len(cand_i), desc="Scoring candidate pairs", unit="pair"):
                vals = [v for v in (prot, sh, ident, ident_sh) if v == v]  # drop NaN
                if vals and max(vals) >= self.edge_floor:
                    records.append((id1, id2, prot, sh))
        print(f"[info] kept {len(records):,} edges (best metric >= {self.edge_floor})")

        # Sparse edge list: everything needed to cluster at any metric/threshold >= floor.
        edges = pd.DataFrame(records, columns=['id1', 'id2', 'seq-sim', 'pocket-sim'])
        edges.to_parquet(f'{DATA_DIR}/metadata/crown_pair_similarity.parquet', index=False)
        with open(f'{DATA_DIR}/metadata/crown_complex_ids.json', 'w') as fh:
            json.dump(complex_ids, fh)  # full node list, needed to recover singletons

        # Minimum spanning forest per metric = the single-linkage merge tree. Each is at
        # most n_complexes - 1 rows, so all four together are tiny, and they let you
        # relabel at ANY threshold >= edge_floor without recomputing pairwise scores.
        metrics = ['seq-sim', 'pocket-sim']
        forests = []
        for m in metrics:
            forest = self._build_mst(edges, complex_ids, m)
            forest.insert(0, 'metric', m)
            forests.append(forest)
        mst = pd.concat(forests, ignore_index=True)
        mst.to_parquet(f'{DATA_DIR}/metadata/crown_mst.parquet', index=False)

        data_dict = {}
        data_dict['basename'] = complex_ids

        for cluster_metric in metrics:
            for cluster_threshold in [0.5, 0.7, 0.9]:
                col_name = f'{cluster_threshold} {cluster_metric} cluster'
                labels = self._labels_from_mst(mst[mst['metric'] == cluster_metric], complex_ids, cluster_threshold)
                data_dict[col_name] = labels

        df = pd.DataFrame(data_dict)
        df.to_csv(f'{DATA_DIR}/metadata/crown_seq_cluster_labels.csv', index = False)

def cluster_mmseqs():
    FastaBuilder().main()
    SequenceComparer().main()
    cleanup_mmseqs()
