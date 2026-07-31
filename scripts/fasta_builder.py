from src.config import DATA_DIR

import os
import json
import numpy as np
import gemmi
from scipy.spatial import KDTree
from biopandas.mol2 import PandasMol2
from collections import defaultdict
from tqdm import tqdm


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
        """Write one FASTA file per chain to {DATA_DIR}/fasta_files/{subdir}_{chain}.fasta"""
        fasta_dir = f'{DATA_DIR}/fasta_files'
        os.makedirs(fasta_dir, exist_ok=True)

        for chain_name, seq in sequences.items():
            path = f'{fasta_dir}/{subdir}_{chain_name}.fasta'
            with open(path, 'w') as fh:
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
        out_path = f'{DATA_DIR}/binding_residues.json'
        with open(out_path, 'w') as fh:
            json.dump(binding_residues_dict, fh, indent=2)


if __name__ == '__main__':
    FastaBuilder().main()