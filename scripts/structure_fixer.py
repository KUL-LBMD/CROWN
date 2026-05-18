from src.config import DATA_DIR
from src.CROWN.ccd_cache import _load_ccd_cache
from src.CROWN.utils import remove_artifacts_and_fix_quotes

# Main structural biology dependencies
import gemmi
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Modeller

# Basic resources
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from itertools import product
import traceback

# File management
import os
from joblib import Parallel, delayed

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ["OPENMM_DEFAULT_PLATFORM"] = "CPU"
os.environ["OPENMM_CPU_THREADS"] = "1"

### Define important variables ###
#---------------------------------

STANDARD_AA = {'ALA', 'ARG', 'ASH', 'ASN', 'ASP', 'CYM', 'CYS', 'CYX', 'GLH', 'GLN', 'GLU', 'GLY', 'HIS', 'HID', 'HIE', 'HIP', 'HYP', 'ILE', 'LEU', 'LYN', 
               'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'CALA', 'CARG', 'CASN', 'CASP', 'CCYS', 'CCYX', 'CGLN', 'CGLU', 'CGLY', 
               'CHID', 'CHIE', 'CHIP', 'CHYP', 'CILE', 'CLEU', 'CLYS', 'CMET', 'CPHE', 'CPRO', 'CSER', 'CTHR', 'CTRP', 'CTYR', 'CVAL', 'NHE', 'NME', 
               'ACE', 'NALA', 'NARG', 'NASN', 'NASP', 'NCYS', 'NCYX', 'NGLN', 'NGLU', 'NGLY', 'NHID', 'NHIE', 'NHIP', 'NILE', 'NLEU', 'NLYS', 'NMET', 
               'NPHE', 'NPRO', 'NSER', 'NTHR', 'NTRP', 'NTYR', 'NVAL', 'HSD', 'HSE', 'HSP'}

DNA_BASES = {'DA', 'DT', 'DG', 'DC'}
RNA_BASES = {'A', 'U', 'G', 'C'}
STANDARD_BASES = DNA_BASES | RNA_BASES

# Sugar–phosphate backbone atoms shared by DNA and RNA (O2' present in RNA only).
# OP1/OP2/OP3 are the canonical PDB names; O1P/O2P/O3P are kept for older files.
RIBOPHOSPHATE_BACKBONE = {
    'P', 'OP1', 'OP2', 'OP3', 'O1P', 'O2P', 'O3P',
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "O2'",
}

VALID_BOND_ATOMS = {'C', 'N', 'O', 'S', 'P', 'B'}

WATER_NAMES = {"HOH", "WAT", "DOD", "H2O", "TIP3", "SPC", "TIP"}
METALLOCOFACTORS = {'HEM', 'SF4', 'MGD'}
FAILURE_RESIDUES = {'Xe', 'W', 'OXY', 'CMO', 'AM', 'UNK', 'UNX', 'UNL'}

FIXED_RESIDUES = STANDARD_AA | STANDARD_BASES | METALLOCOFACTORS | WATER_NAMES | {'ACE', 'NME'}
CUSTOM_SUBSTITUTIONS = {'SEC': 'CYS', '0A8': 'CYS', 'MSE': 'MET'}

CLASH_RADIUS = 1.8
CHAIN_RADIUS = 4.0
SHELL_RADIUS = 6.0

MAX_TERMINAL_EXTENSION = 3

### Helper functions ###
#-----------------------

def normalize(s: str) -> str:
    """Canonicalize a CCD atom name.

    Removes stray backslashes and strips CIF-style surrounding quotes
    (a matching pair of `"` or `'` wrapping the whole token). A trailing
    prime — meaningful in nucleotide names like `O5'` — is preserved.
    """
    s = s.replace("\\", "")
    if len(s) >= 2 and s[0] in ('"', "'") and s[0] == s[-1]:
        s = s[1:-1]
    return s

def _remove_chain(model: gemmi.Model, name: str) -> None:
    for i in range(len(model) - 1, -1, -1):
        if model[i].name == name:
            del model[i]


def _has_chain(model: gemmi.Model, name: str) -> bool:
    return any(c.name == name for c in model)

def _count_heavy_atoms(model: gemmi.Model, name: str):

    chain = model[name]
    heavy_atom_count = 0

    for residue in chain:
        for atom in residue:
            if atom.element.name in VALID_BOND_ATOMS:
                heavy_atom_count += 1

    return heavy_atom_count

def _has_contact(chain: gemmi.Chain, target_tree: KDTree):
    """
    Checks whether chain is in close proximity to target chain
    """
    coords = []
    for res in chain:
        for atom in res:
            if not atom.element.name in {'H', 'D'}:
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])

    if not coords:
        return False
    
    dists, _ = target_tree.query(np.array(coords))
    return bool(np.any(dists < CHAIN_RADIUS))

def _apply_sym_op(structure: gemmi.Structure, chain: gemmi.Chain, op: gemmi.Op, du: int, dv: int, dw: int) -> gemmi.Chain:
    """
    Create new chain through symmetry operation
    """

    cell = structure.cell

    clone = chain.clone()
    for res in clone:
        for atom in res:
            frac = cell.fractionalize(atom.pos)
            tf = op.apply_to_xyz([frac.x, frac.y, frac.z])
            atom.pos = cell.orthogonalize(gemmi.Fractional(tf[0] + du, tf[1] + dv, tf[2] + dw))

    return clone

def _setup_source_entities(structure: gemmi.Structure):
    """Ensure source structure has entities set up."""
    structure.setup_entities()
    structure.assign_subchains()


def _map_chains_to_entities(structure: gemmi.Structure) -> dict:
    """Map chain name -> Entity using the subchain relationship."""
    sub_to_entity = {}
    for ent in structure.entities:
        for sub_id in ent.subchains:
            sub_to_entity[sub_id] = ent

    chain_to_entity = {}
    for chain in structure[0]:
        for sub_span in chain.subchains():
            sub_id = sub_span[0].subchain if len(sub_span) > 0 else None
            if sub_id and sub_id in sub_to_entity:
                chain_to_entity[chain.name] = sub_to_entity[sub_id]
                break

    return chain_to_entity


def _assign_subchain_ids(structure: gemmi.Structure):
    """
    Set each residue's subchain to match its chain name,
    so that entity.subchains references resolve correctly.
    """
    for chain in structure[0]:
        for residue in chain:
            residue.subchain = chain.name

def _get_chain_coords(structure: gemmi.Structure, chain_id: str):

    coords = []

    model = structure[0]
    chain = model[chain_id]
    for residue in chain:
        for atom in residue:
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])

    return np.array(coords)

def update_element_positions(input_path):

    line_list = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78]
                shifted_element = line[75:77]
                if len(element.strip()) == 1 and len(shifted_element.strip()) == 2:
                    line = (line[:75] + ' ' + shifted_element.ljust(2) + line[78:])

            line_list.append(line)

    with open(input_path, 'w') as f:
        f.write('\n'.join(line_list))

def is_nonstandard_residue(residue, chain_residues):
    """
    Nonstandard residues should have all required backbone elements and should be flanked by standard residues
    """

    elements = {atom.element.symbol for atom in residue.atoms()}
    if {'C', 'N', 'O'}.issubset(elements):
         # Must be flanked by standard residues
        res_list = chain_residues[residue.chain.index]
        local_idx = next(i for i, r in enumerate(res_list) if r == residue)
        if local_idx != 0 and local_idx != len(res_list) - 1:
            prev_res = res_list[local_idx - 1]
            next_res = res_list[local_idx + 1]
            if prev_res.name in FIXED_RESIDUES or next_res.name in FIXED_RESIDUES:
                return True

        elif local_idx == 0 and len(res_list) > 1:
            next_res = res_list[local_idx + 1]
            if next_res.name in FIXED_RESIDUES:
                return True

        elif local_idx == len(res_list) - 1 and len(res_list) > 1:
            prev_res = res_list[local_idx - 1]
            if prev_res.name in FIXED_RESIDUES:
                return True

    return False

@dataclass
class ResidueContact:
    """Store information about contacts between residues"""
    chain1: str
    res1: int
    chain2: str
    res2: int
    contact_count: int
    atom_pairs: List[Tuple]

def _topological_sort(nodes: Set[str], edges: List[Tuple[str, str]]) -> List[str]:
    """Kahn's algorithm with alphabetical tie-breaking."""
    import bisect
    from collections import defaultdict

    in_degree = {n: 0 for n in nodes}
    adj = defaultdict(list)
    for u, v in edges:
        if u in in_degree and v in in_degree:
            adj[u].append(v)
            in_degree[v] += 1

    available = sorted(n for n in nodes if in_degree[n] == 0)
    result: List[str] = []
    while available:
        node = available.pop(0)
        result.append(node)
        for nxt in sorted(adj[node]):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                bisect.insort(available, nxt)

    # cycles / leftovers — append alphabetically so we never crash
    if len(result) != len(nodes):
        for n in sorted(nodes):
            if n not in result:
                result.append(n)
    return result

### Core functions ###
#----------------

def select_conformer(structure: gemmi.Structure):
    """
    Reduce residues to maximum atoms / maximum occupancy conformers.

    Parameters
    ----------
    structure: gemmi.Structure
    """

    for model in structure:
        for chain in model:
            for residue in chain:
                # Collect altloc groups: '' (no altloc), 'A', 'B', etc.
                groups: defaultdict[str, list[gemmi.Atom]] = defaultdict(list)
                for atom in residue:
                    groups[atom.altloc].append(atom)

                # If no alternates exist, nothing to do
                altlocs = [k for k in groups if k != '\x00']  # '\x00' is gemmi's "no altloc"
                if not altlocs:
                    continue

                # Pick winner: most atoms first, then highest mean occupancy
                def score(altloc: str) -> tuple[int, float]:
                    atoms = groups[altloc]
                    return (len(atoms), sum(a.occ for a in atoms) / len(atoms))

                winner = max(altlocs, key=score)
                loser_altlocs = set(altlocs) - {winner}

                # Delete losing atoms in reverse to preserve indices
                for i in range(len(residue) - 1, -1, -1):
                    if residue[i].altloc in loser_altlocs:
                        del residue[i]

                # Clear altloc labels on remaining atoms
                for atom in residue:
                    atom.altloc = '\x00'

    # Collapse residue-level microheterogeneity:
    # if a chain has multiple residues sharing the same (seqid_num, icode),
    # keep the one with the most heavy atoms (tiebreak: mean occupancy).
    for model in structure:
        for chain in model:
            groups: defaultdict[tuple, list[int]] = defaultdict(list)
            for i, res in enumerate(chain):
                groups[(res.seqid.num, res.seqid.icode)].append(i)

            to_remove: list[int] = []
            for key, indices in groups.items():
                if len(indices) == 1:
                    continue

                def score(i: int) -> tuple[int, float]:
                    res = chain[i]
                    heavy = [a for a in res if not a.is_hydrogen()]
                    if not heavy:
                        return (0, 0.0)
                    return (len(heavy), sum(a.occ for a in heavy) / len(heavy))

                winner = max(indices, key=score)
                to_remove.extend(i for i in indices if i != winner)

            for i in sorted(to_remove, reverse=True):
                del chain[i]

    return structure

class OverlapResolver:
    """
    Detect and resolve missing bonds and steric clashes between chains
    """

    def __init__(self):
        pass

    def detect_contacts(self, structure: gemmi.Structure):
        """
        Detects all contacts between residues in different chains

        Parameters
        ----------
        structure [gemmi.Structure]

        Returns
        -------
        Tuple of:
            - Dictionary of contacts (key: sorted chain pair, value: ResidueContact)
            - List of chain pairs with 1 contact (bonds to add)
            - List of chain pairs with multiple contacts (overlaps to resolve)
        """

        model = structure[0]

        # Single search tree at the larger radius; we query sub-radii later
        ns = gemmi.NeighborSearch(model, structure.cell, CLASH_RADIUS)
        ns.populate(include_h=False)

        contact_counts: Dict[tuple, int] = defaultdict(int)
        contact_details: Dict[tuple, list] = defaultdict(list)

        for chain in model:
            for residue in chain:
                if residue.name in WATER_NAMES:
                    continue
                for atom in residue:
                    if atom.element.name not in VALID_BOND_ATOMS:
                        continue

                    chain1_id = chain.name
                    res1_id = residue.seqid.num
                    res1_name = residue.name

                    # ── Inter-chain contacts ──
                    for mark in ns.find_neighbors(atom, max_dist = CLASH_RADIUS):
                        cra = mark.to_cra(model) # Chain, Residue, Atom
                        if cra.residue.name in WATER_NAMES or cra.atom.element.name not in VALID_BOND_ATOMS:
                            continue

                        chain2_id = cra.chain.name
                        res2_id = cra.residue.seqid.num

                        if chain1_id != chain2_id:
                            pair_key = tuple(sorted([chain1_id, chain2_id]))
                            atom_pair = tuple(sorted([
                                (chain1_id, res1_id, atom.name),
                                (chain2_id, res2_id, cra.atom.name),
                            ]))
                            if atom_pair not in contact_details[pair_key]:
                                contact_counts[pair_key] += 1
                                contact_details[pair_key].append(atom_pair)

        # ── Classify contacts ──
        bonds_to_add = []
        overlaps_to_resolve = []
        contact_info = {}

        for pair_key, count in contact_counts.items():
            chain1, chain2 = pair_key
            contact_info[pair_key] = ResidueContact(
                chain1=chain1,
                chain2=chain2,
                res1=None,
                res2=None,
                contact_count=count,
                atom_pairs=contact_details[pair_key],
            )
            if count == 1:
                bonds_to_add.append(f"{chain1},{chain2}")
            else:
                overlaps_to_resolve.append(f"{chain1},{chain2}")

        return contact_info, bonds_to_add, overlaps_to_resolve

    def resolve_overlaps(self, structure, overlaps, bonds=None):
        model = structure[0]
        removed = set()
        for overlap_pair in overlaps:
            chain1_id, chain2_id = overlap_pair.split(",")
            if _has_chain(model, chain1_id) and _has_chain(model, chain2_id):
                if _count_heavy_atoms(model, chain1_id) > _count_heavy_atoms(model, chain2_id):
                    _remove_chain(model, chain2_id); removed.add(chain2_id)
                else:
                    _remove_chain(model, chain1_id); removed.add(chain1_id)
        if bonds is not None:
            return [b for b in bonds if not (set(b.split(",")) & removed)]

    def merge_bonded_chains(self, structure: gemmi.Structure, bonds: List[str], contact_info):
        """
        Merge chains and rename to alphabetically first chain.

        Protein chains are merged in C→N order (the chain whose C-terminus forms
        the peptide bond comes first). DNA chains are merged in O3'→P order.
        Branch chains are appended at the end of the merged chain.
        """
        model = structure[0]

        # ------------------------------------------------------------------
        # 1. Classify every chain we'll touch
        # ------------------------------------------------------------------
        chain_class: Dict[str, str] = {}

        def classify(chain_id: str) -> str:
            chain = model[chain_id]
            res_names = [residue.name for residue in chain]
            standard_aa = sum(1 for r in res_names if r in STANDARD_AA)
            standard_bases = sum(1 for r in res_names if r in STANDARD_BASES)
            if standard_aa > 5:
                return 'protein'
            if standard_bases > 4:
                return 'dna'
            return 'branch'

        # ------------------------------------------------------------------
        # 2. First pass: validate bonds + record directed backbone edges
        #    Edge (u, v)  means  u's residues come BEFORE v's residues.
        # ------------------------------------------------------------------
        cleaned_bonds: List[str] = []
        backbone_edges: List[Tuple[str, str]] = []

        for bond in bonds:
            chain1_id, chain2_id = bond.split(',')
            pair_key = tuple(sorted([chain1_id, chain2_id]))
            pair_contact_info = contact_info[pair_key]

            for cid in (chain1_id, chain2_id):
                if cid not in chain_class:
                    chain_class[cid] = classify(cid)

            chain_classes = {chain_class[chain1_id], chain_class[chain2_id]}

            # atom_pair entries are (chain_id, residue_seqid, atom_name)
            atom1, atom2 = pair_contact_info.atom_pairs[0]
            a1_chain, a1_name = atom1[0], atom1[2]
            a2_chain, a2_name = atom2[0], atom2[2]
            atom_names_sorted = sorted([a1_name, a2_name])

            if 'branch' in chain_classes:
                # branch contacts are kept; no direction (they go to the end)
                cleaned_bonds.append(bond)

            elif chain_classes == {'protein'}:
                if atom_names_sorted == ['C', 'N']:
                    cleaned_bonds.append(bond)
                    # C-side chain comes before N-side chain
                    if a1_name == 'C':
                        backbone_edges.append((a1_chain, a2_chain))
                    else:
                        backbone_edges.append((a2_chain, a1_chain))

            elif chain_classes == {'dna'}:
                if atom_names_sorted == ["O3'", 'P']:
                    cleaned_bonds.append(bond)
                    # 3'-side chain comes before 5'-side chain
                    if a1_name == "O3'":
                        backbone_edges.append((a1_chain, a2_chain))
                    else:
                        backbone_edges.append((a2_chain, a1_chain))

            # Protein-DNA: alarm!
            else:
                return 'alarm'

        # ------------------------------------------------------------------
        # 3. Connected components from the *validated* bonds
        # ------------------------------------------------------------------
        chain_sets = [set(b.split(",")) for b in cleaned_bonds]
        merged = True
        while merged:
            merged = False
            new_sets = []
            while chain_sets:
                current = chain_sets.pop(0)
                for other in chain_sets[:]:
                    if current & other:
                        current |= other
                        chain_sets.remove(other)
                        merged = True
                new_sets.append(current)
            chain_sets = new_sets

        # ------------------------------------------------------------------
        # 4. Per component: backbone (topo-sorted) + branches (alphabetical)
        # ------------------------------------------------------------------
        for chain_set in chain_sets:
            backbone_chains = {c for c in chain_set if chain_class.get(c) != 'branch'}
            branch_chains   = sorted(c for c in chain_set if chain_class.get(c) == 'branch')

            component_edges = [(u, v) for (u, v) in backbone_edges
                            if u in backbone_chains and v in backbone_chains]

            ordered_backbone = _topological_sort(backbone_chains, component_edges)
            ordered_chains   = ordered_backbone + branch_chains

            new_chain_name = sorted(chain_set)[0]

            all_residues = []
            chains_to_remove = []
            for chain_id in ordered_chains:
                if not _has_chain(model, chain_id):
                    continue
                chain = model[chain_id]
                for residue in chain:
                    all_residues.append(residue.clone())
                chains_to_remove.append(chain_id)

            if not all_residues:
                continue  # was `return` in the original — that aborted later components

            for chain_id in chains_to_remove:
                _remove_chain(model, chain_id)

            new_chain = gemmi.Chain(new_chain_name)
            for new_index, residue in enumerate(all_residues, start=1):
                residue.seqid = gemmi.SeqId(str(new_index))
                new_chain.add_residue(residue)
            model.add_chain(new_chain)

        return 'ok'

def select_ligand_chains(structure: gemmi.Structure):
    """
    Select possible ligand chains: 10-100 heavy atoms, at least 1 C, no metals (except for METALLOCOFACTORS)

    Parameters
    ----------
    structure [gemmi.Structure]

    Returns
    -------
    ligand_chains [List[str]]
    """

    ligand_chains = []

    model = structure[0]
    for chain in model:
        chain_id = chain.name

        # 3 selection criteria
        atom_count = 0
        num_carbons = 0
        no_metal_bool = True

        for residue in chain:
            res_name = residue.name
            for atom in residue:
                element = atom.element.name

                if element in {'H', 'D'}:
                    continue

                if element == 'C':
                    num_carbons += 1

                if element in {'C', 'N', 'O', 'S', 'P', 'B', 'F', 'Cl', 'Br', 'I'}:
                    atom_count += 1
                elif not res_name in METALLOCOFACTORS:
                    no_metal_bool = False

        allowed_length_bool = atom_count >= 10 and atom_count <= 100
        carbon_bool = num_carbons > 2

        if allowed_length_bool and carbon_bool and no_metal_bool:
            ligand_chains.append(chain_id)

    return ligand_chains

def detect_unresolved_ligand_atoms(structure: gemmi.Structure, chain_id: str):
    """
    Detect unresolved heavy atoms by matching with mmCIF components template

    Parameters
    ----------
    structure [gemmi.Structure]
    chain_id [str]

    Returns
    -------
    fully_resolved [bool]
    """

    ccd = _load_ccd_cache()
    chain = structure[0][chain_id]

    for residue in chain:
        expected = ccd.get(residue.name)
        if expected is None:
            print(f'Warning: CCD code {residue.name} not found in cache')
            return False

        heavy_atoms = [atom for atom in residue if atom.element.name not in {'H', 'D'}]

        # Check occupancy for all heavy atoms
        zero_occ = [atom.name for atom in heavy_atoms if atom.occ <= 0.00]
        if zero_occ:
            print(f'Warning: zero occupancy heavy atoms in {residue.name}: {zero_occ}')
            return False

        observed = {normalize(atom.name) for atom in residue if atom.element.name not in {'H', 'D'}}

        absent = expected - observed
        absent = [x[0] for x in absent]

        if absent:
            if len(absent) > 1 or absent[0] != 'O':
                return False

    return True

def build_pdb(structure: gemmi.Structure, chain_id):
    """
    1. Applies symmetry expansion
    2. Selects only neighboring chains
    3. Removes residues from symmetry-mates with heavy-atom clashes (< 1 Å)
       against already-placed atoms
    4. Renames chains and updates SEQRES
    5. Writes as PDB
    """
    CLASH_CUTOFF = 1.0  # Å, heavy-atom contact threshold

    model = structure[0]
    cell = structure.cell
    sg = structure.find_spacegroup()
    target_chain = model[chain_id]

    # Target heavy-atom coords → seed of the "existing" set
    target_coords = [
        [atom.pos.x, atom.pos.y, atom.pos.z]
        for res in target_chain
        for atom in res
        if atom.element.name not in {'H', 'D'}
    ]
    target_tree = KDTree(np.array(target_coords))

    # Symmetry expansion and neighbor selection (unchanged)
    neighbors = []
    for chain in model:
        if sg is not None:
            for op in sg.operations():
                for du, dv, dw in product(range(-1, 2), repeat=3):
                    is_self = (
                        op.triplet() == "x,y,z"
                        and (du, dv, dw) == (0, 0, 0)
                        and chain.name == chain_id
                    )
                    if is_self:
                        continue
                    transformed = _apply_sym_op(structure, chain, op, du, dv, dw)
                    if _has_contact(transformed, target_tree):
                        neighbors.append((transformed, chain.name))
        else:
            if chain.name != chain_id:
                if _has_contact(chain, target_tree):
                    neighbors.append((chain.clone(), chain.name))

    # Build output structure
    new_structure = gemmi.Structure()
    new_structure.cell = cell
    new_structure.spacegroup_hm = 'P 1'
    new_model = gemmi.Model(0)

    # Target chain → Z
    target_clone = target_chain.clone()
    target_clone.name = "Z"
    new_model.add_chain(target_clone)

    # Running set of heavy-atom coords already placed in the output
    existing_coords = list(target_coords)

    # Neighbors → A, B, C, … with residue-level clash filtering
    available = list("ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz0123456789")
    source_map = {"Z": chain_id}
    out_idx = 0

    for nchain, src_name in neighbors:
        # KDTree over everything placed so far
        existing_tree = KDTree(np.array(existing_coords))

        # Flag clashing residues by index
        clashing = set()
        for i, res in enumerate(nchain):
            res_coords = np.array([
                [a.pos.x, a.pos.y, a.pos.z]
                for a in res if a.element.name not in {'H', 'D'}
            ])
            if res_coords.size == 0:
                continue
            dists, _ = existing_tree.query(res_coords, k=1)
            if np.any(dists < CLASH_CUTOFF):
                clashing.add(i)

        # Rebuild the chain without clashing residues
        if clashing:
            filtered = gemmi.Chain(nchain.name)
            for i, res in enumerate(nchain):
                if i not in clashing:
                    filtered.add_residue(res)
            nchain = filtered

        # Skip if nothing survives
        if len(nchain) == 0:
            continue

        # Accept the chain
        new_name = available[out_idx]
        out_idx += 1
        nchain.name = new_name
        new_model.add_chain(nchain)
        source_map[new_name] = src_name

        # Extend the existing-atoms pool with this chain's heavy atoms
        for res in nchain:
            for a in res:
                if a.element.name not in {'H', 'D'}:
                    existing_coords.append([a.pos.x, a.pos.y, a.pos.z])

    new_structure.add_model(new_model)

    # ── Rebuild entities for SEQRES (unchanged) ──
    _setup_source_entities(structure)
    src_entity_for_chain = _map_chains_to_entities(structure)
    for new_chain in new_structure[0]:
        src_chain_name = source_map.get(new_chain.name)
        if src_chain_name is None:
            continue
        src_entity = src_entity_for_chain.get(src_chain_name)
        if src_entity is None or src_entity.entity_type != gemmi.EntityType.Polymer:
            continue
        ent = gemmi.Entity(new_chain.name)
        ent.entity_type = src_entity.entity_type
        ent.polymer_type = src_entity.polymer_type
        ent.full_sequence = src_entity.full_sequence
        ent.subchains = [new_chain.name]
        new_structure.entities.append(ent)

    _assign_subchain_ids(new_structure)
    return new_structure

def fix_missing_and_nonstandard_residues(basename: str, ligand_coords: np.ndarray):
    """
    Use PDBFixer to find nonstandard residues, missing residues and unresolved atoms.

    If **all** nonstandard residues are farther than *distance_cutoff* Å from
	every atom in *ligand_coords*, they are silently replaced by their standard
	counterparts and the PDB is overwritten in-place.

	If any nonstandard residue has at least one atom within *distance_cutoff* Å
	of a ligand atom, the file is left untouched and the function returns
	``False`` so the caller can abort.
    """

    fixer = PDBFixer(filename = f'{DATA_DIR}/pdb/raw/{basename}.pdb')

    # ── Branched residues merged into polymer chains ──
    # After merge_bonded_chains, glycans (NAG/BMA/MAN/…) and other covalent
    # attachments end up on the protein chain. Detect them as non-FIXED
    # residues lacking a CA backbone atom inside a long (>100 heavy atom)
    # chain. Drop them if they sit outside the binding-site shell; abort if
    # any of their atoms fall inside it.
    chain_heavy_counts = {}
    chain_id_map = {}
    for chain in fixer.topology.chains():
        heavy = 0
        for residue in chain.residues():
            chain_id_map[residue.index] = chain.id
            for atom in residue.atoms():
                if atom.element is not None and atom.element.symbol in VALID_BOND_ATOMS:
                    heavy += 1
        chain_heavy_counts[chain.index] = heavy

    nucleic_atoms_to_strip = []
    branched = []
    for chain in fixer.topology.chains():
        if chain.id != 'Z' and chain_heavy_counts.get(chain.index, 0) > 100:
            residues = list(chain.residues())
            n_dna = sum(1 for r in residues if r.name in DNA_BASES)
            n_rna = sum(1 for r in residues if r.name in RNA_BASES)
            n_res = len(residues)

            # Deal with DNA chains: terminal residues or mutated residues?
            if n_dna > 2 or n_rna > 2:
                replacement_base = 'DA' if n_dna >= n_rna else 'A'
                for idx, residue in enumerate(residues):
                    if not residue.name in FIXED_RESIDUES:
                        if idx == 0 or idx == n_res - 1:
                            branched.append(residue)
                        else:
                            residue.name = replacement_base
                            for atom in residue.atoms():
                                if atom.name not in RIBOPHOSPHATE_BACKBONE:
                                    nucleic_atoms_to_strip.append(atom)

            else:
                for idx, residue in enumerate(residues):
                    if not residue.name in FIXED_RESIDUES:
                        if idx == 0 or idx == n_res - 1:
                            branched.append(residue)

    for residue in branched:
        for atom in residue.atoms():
            pos = fixer.positions[atom.index]
            atom_coord = np.array([pos.x * 10.0, pos.y * 10.0, pos.z * 10.0])
            if np.linalg.norm(ligand_coords - atom_coord, axis=1).min() < SHELL_RADIUS:
                print(
                    f'Branched residue in shell: '
                    f'{basename} {residue.name} {residue.chain.id}{residue.id}'
                )
                return 'branched_in_shell'

    if branched:
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(branched)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

    # Remove distal water residues
    waters_to_remove = []
    for residue in fixer.topology.residues():
        if residue.name in WATER_NAMES:
            residue.name = 'HOH'
            atom = next(residue.atoms())
            pos = fixer.positions[atom.index]
            atom_coord = np.array([pos.x * 10.0, pos.y * 10.0, pos.z * 10.0])
            if np.linalg.norm(ligand_coords - atom_coord, axis=1).min() > SHELL_RADIUS:
                waters_to_remove.append(residue)

    if waters_to_remove:
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(waters_to_remove)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

    # Remove H or D atoms
    atoms_to_remove = []
    for residue in fixer.topology.residues():
        for atom in residue.atoms():
            if atom.element.atomic_number == 1:
                atoms_to_remove.append(atom)

    if atoms_to_remove:
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(atoms_to_remove)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

    # ── topology may have changed; rebuild the per-chain bookkeeping ──
    chain_residues = {}
    chain_id_map = {}
    chain_heavy_counts = {}
    for chain in fixer.topology.chains():
        chain_residues[chain.index] = list(chain.residues())
        heavy = 0
        for residue in chain.residues():
            chain_id_map[residue.index] = chain.id
            for atom in residue.atoms():
                if atom.element is not None and atom.element.symbol in VALID_BOND_ATOMS:
                    heavy += 1
        chain_heavy_counts[chain.index] = heavy

    fixer.findNonstandardResidues()

    # Apply custom substitutions and default-to-ALA fallback
    already_flagged = {r for r, _ in fixer.nonstandardResidues}

    # Remove any chain Z residues that PDBFixer auto-flagged
    fixer.nonstandardResidues = [
        (r, sub) for r, sub in fixer.nonstandardResidues
        if chain_heavy_counts.get(r.chain.index, 0) > 100
    ]  

    for residue in fixer.topology.residues():
        if chain_id_map.get(residue.index) == "Z":
            continue

        elements = {atom.element for atom in residue.atoms()}
        if None in elements:
            return

        if residue.name not in FIXED_RESIDUES and residue not in already_flagged:
            if chain_heavy_counts.get(residue.chain.index, 0) <= 100:
                continue
            if is_nonstandard_residue(residue, chain_residues):
                replacement = CUSTOM_SUBSTITUTIONS.get(residue.name, "ALA")
                fixer.nonstandardResidues.append((residue, replacement))

    # Drop small-chain nonstandard residues from consideration entirely.
    # Chains with ≤100 heavy atoms are treated as peptide ligands/fragments
    # where residues like ABA/SAR are legitimate building blocks, not
    # modifications to be replaced with ALA.
    fixer.nonstandardResidues = [
        (r, sub) for r, sub in fixer.nonstandardResidues
        if chain_heavy_counts.get(r.chain.index, 0) > 100
    ]

    n_pos = len(fixer.positions)

    # Check each remaining nonstandard residue for proximity to any ligand atom
    if fixer.nonstandardResidues:
        for residue, _replacement in fixer.nonstandardResidues:
            for atom in residue.atoms():

                if atom.index >= n_pos:
                    continue

                pos = fixer.positions[atom.index]
                atom_coord = np.array([pos.x * 10.0, pos.y * 10.0, pos.z * 10.0])
                dists = np.linalg.norm(ligand_coords - atom_coord, axis=1)
                if dists.min() < SHELL_RADIUS:
                    print(
                        f'Modified residue in shell: '
                        f'{basename} {residue.name} {residue.chain.id}{residue.id}'
                    )
                    return 'modified_in_shell'
        # All remaining nonstandard residues are far from ligands – safe to replace
        fixer.replaceNonstandardResidues()

    fixer.findMissingResidues()
    fixer.findMissingAtoms()

    # Exclude chain Z from missing residues
    chain_idx_to_id = {c.index: c.id for c in fixer.topology.chains()}

    # Cap terminal extensions at 5 residues. PDBFixer's missingResidues key is
    # (chain_index, insertion_position), where the position is an index *into
    # the existing residues* — 0 means insert before the first, len(chain) means
    # append after the last. Internal gaps are left untouched.
    chain_res_counts = {idx: len(res_list) for idx, res_list in chain_residues.items()}

    for (chain_idx, ins_pos), residues in list(fixer.missingResidues.items()):
        if len(residues) <= MAX_TERMINAL_EXTENSION:
            continue
        if ins_pos == 0:
            # N-terminal: list runs from SEQRES start toward the existing chain,
            # so the residues nearest the resolved structure are at the end.
            fixer.missingResidues[(chain_idx, ins_pos)] = residues[-MAX_TERMINAL_EXTENSION:]
        elif ins_pos == chain_res_counts.get(chain_idx, 0):
            # C-terminal: list runs from existing chain outward, keep the head.
            fixer.missingResidues[(chain_idx, ins_pos)] = residues[:MAX_TERMINAL_EXTENSION]
        # else: internal gap, leave it alone

    n_pos = len(fixer.positions)

    for (chain_idx, ins_pos), residues in fixer.missingResidues.items():
        n_chain = chain_res_counts.get(chain_idx, 0)
        if ins_pos == 0:
            flanks = [chain_residues[chain_idx][ins_pos]]
        elif ins_pos == n_chain:
            flanks = [chain_residues[chain_idx][ins_pos - 1]]
        else:
            flanks = [chain_residues[chain_idx][ins_pos - 1], chain_residues[chain_idx][ins_pos]]
        for res in flanks:
            for atom in res.atoms():
                if atom.index >= n_pos:
                    continue
                pos = fixer.positions[atom.index]
                coord = np.array([pos.x * 10.0, pos.y * 10.0, pos.z * 10.0])
                if np.linalg.norm(ligand_coords - coord, axis=1).min() < SHELL_RADIUS:
                    print(f'{basename} - Gap in shell')
                    return 'gap_in_shell'

    # Check if any remaining residue with missing atoms is near the ligand
    n_pos = len(fixer.positions)

    for residue in list(fixer.missingAtoms.keys()):

        missing = fixer.missingAtoms[residue]
        # Skip the shell check if the only missing atom is OXT.
        if len(missing) == 1 and missing[0].name == 'OXT':
            continue

        for atom in residue.atoms():
            if atom.index >= n_pos:
                continue
            pos = fixer.positions[atom.index]
            atom_coord = np.array([pos.x * 10.0, pos.y * 10.0, pos.z * 10.0])
            dists = np.linalg.norm(ligand_coords - atom_coord, axis=1)
            if dists.min() < SHELL_RADIUS:
                print(
                    f'Missing residue in shell: '
                    f'{basename} {residue.name} {residue.chain.id}{residue.id}'
                )
                return 'missing_atoms_in_shell'

    # Impossible to build missing non-standard residues: switch these to ALA instead
    for key, value in fixer.missingResidues.items():
        # value is a list of resnames: if any entry is non-standard, replace it with ALA
        for i in range(len(value)):
            if not value[i] in FIXED_RESIDUES:
                value[i] = 'ALA'

    fixer.addMissingAtoms()

    # ── Strip stray OXT atoms from non-C-terminal residues ──
    # OXT is only valid on the very last residue of a polypeptide chain;
    # PDBFixer occasionally leaves them on internal residues after
    # merge/extension steps. Drop them so downstream parsers don't
    # interpret them as chain breaks.
    oxt_to_remove = []
    for chain in fixer.topology.chains():
        residues = list(chain.residues())
        if len(residues) <= 1:
            continue
        for residue in residues[:-1]:
            for atom in residue.atoms():
                if atom.name == 'OXT':
                    oxt_to_remove.append(atom)
 
    if oxt_to_remove:
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(oxt_to_remove)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions


    with open(f'{DATA_DIR}/pdb/fixed/{basename}.pdb', "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    return 'ok'

def check_structure(basename):
    """
    Check against failure residues, huge residues and failure elements
    """

    structure = gemmi.read_structure(f'{DATA_DIR}/pdb/fixed/{basename}.pdb')
    model = structure[0]

    for chain in model:
        for residue in chain:
            if residue.name in FAILURE_RESIDUES:
                print(f'{basename} - {residue.name} in structure')
                return False
            
            n_heavy = sum(1 for atom in residue if not atom.is_hydrogen())
            if n_heavy > 90:
                print(f'{basename} - Big motherfucker {residue.name} in structure')
                return False
            
            elements = {atom.element.name for atom in residue}
            if 'C' in elements and not residue.name in METALLOCOFACTORS:
                if not elements.issubset({'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H', 'D'}):
                    print(f'{basename} - {residue.name} in structure')
                    return False
                
    return True

### Core workflow ###
#--------------------

def main(pdb_id):
    """
    Main clean-up of mmCIF files.
    This workflow goes through the following steps:

    1. Reduce residues to maximum atoms / maximum occupancy conformers.
    2. Remove crystallization artifacts
    3. Resolve steric clashes and missing bonds
    4. Select possible ligand chains: 10-100 heavy atoms, at least 1 C, no metals (except for METALLOCOFACTORS)
    5. For each ligand chain:
        5.1. Unresolved ligand atoms?
        5.2. Missing residues / atoms in pocket?
        5.3. Nonstandard residues in pocket?
        5.4. Store as PDB entry

    Parameters
    ----------
    pdb_id [str]: PDB ID of entry 
    """

    flags = {
            'has_missing_atoms_in_shell': False,
            'has_modified_residues_in_shell': False,
            'has_branched_residues_in_shell': False,
            'failure_reason': None,  # None means success
        }

    try:

        # Step 1: remove artifacts and fix file formatting
        structure = gemmi.read_structure(f'{DATA_DIR}/mmCIF/raw/{pdb_id}.cif')
        structure = remove_artifacts_and_fix_quotes(structure)

        # Step 2: reduce conformer
        structure = select_conformer(structure)

        # Step 3: resolve steric clashes and missing bonds between chains
        overlap_resolver = OverlapResolver()
        contact_info, bonds_to_add, overlaps_to_resolve = overlap_resolver.detect_contacts(structure)

        bonds_to_add = overlap_resolver.resolve_overlaps(structure, overlaps_to_resolve, bonds_to_add)
        message = overlap_resolver.merge_bonded_chains(structure, bonds_to_add, contact_info)
        if message == 'alarm':
            return flags

        # 3.2: save cleaned mmCIF structure
        _assign_subchain_ids(structure)
        structure.setup_entities()      # rebuild entity table to match new subchains
        structure.make_mmcif_document().write_file(f'{DATA_DIR}/mmCIF/clean/{pdb_id}.cif')

        # Step 4: select possible ligand chains
        ligand_chains = select_ligand_chains(structure)
        print(f'{pdb_id} - {ligand_chains}')

        # Step 5: loop over ligand chains
        for chain_id in ligand_chains:

            # 5.1: Any unresolved ligand atoms?
            fully_resolved = detect_unresolved_ligand_atoms(structure, chain_id)
            if fully_resolved:

                # 5.2: convert to PDB
                ccd_codes = [res.name for res in structure[0][chain_id]]
                lig_name = '-'.join(ccd_codes)

                new_structure = build_pdb(structure, chain_id)
                contact_info, bonds_to_add, overlaps_to_resolve = overlap_resolver.detect_contacts(new_structure)
                message = overlap_resolver.merge_bonded_chains(new_structure, bonds_to_add, contact_info)
                if message == 'alarm':
                    continue
                basename = f'{pdb_id}_{chain_id}'

                opts = gemmi.PdbWriteOptions()
                opts.ter_ignores_type = True   # only emit TER when the chain name actually changes

                new_structure.write_pdb(f'{DATA_DIR}/pdb/raw/{basename}.pdb', opts)

                # 5.3 missing and nonstand residues with PDBFixer
                update_element_positions(f'{DATA_DIR}/pdb/raw/{basename}.pdb')
                ligand_coords = _get_chain_coords(new_structure, 'Z')

                fixer_status = fix_missing_and_nonstandard_residues(basename, ligand_coords)

                if fixer_status == 'ok':
                    checker_status = check_structure(basename)
                    if not checker_status:
                        os.remove(f'{DATA_DIR}/pdb/fixed/{basename}.pdb')
                
                if fixer_status == 'modified_in_shell':
                    flags['has_modified_residues_in_shell'] = True
                    flags['failure_reason'] = 'modified_residues_in_shell'
                elif fixer_status == 'missing_atoms_in_shell':
                    flags['has_missing_atoms_in_shell'] = True
                    flags['failure_reason'] = 'missing_atoms_in_shell'
                elif fixer_status == 'unknown_residue':
                    flags['failure_reason'] = 'unknown_residue'
                elif fixer_status == 'branched_in_shell':
                    flags['has_branched_residues_in_shell'] = True
                    flags['failure_reason'] = 'branched_residues_in_shell'
                elif fixer_status is None:
                    flags['failure_reason'] = 'null_elements_in_topology'

    except Exception as e:
        print(f'{pdb_id} - {e}')
        traceback.print_exc()
        flags['failure_reason'] = f'exception: {type(e).__name__}: {e}'

    return flags

def wrapper(num_cores = 1):
    """
    Runs through all dataframe entries and converts mmCIF to semi-processed PDB

    Parameters
    ----------

    num_cores [int]: Number of CPU's for parallel processing. Default value = 1
    """

    pdb_list = [x[:4] for x in os.listdir(f'{DATA_DIR}/mmCIF/raw')]
    n_total = len(pdb_list)

    results = Parallel(n_jobs = num_cores, verbose = 10)(delayed(main)(pdb_id) for pdb_id in pdb_list)

    # Aggregate failure reasons
    n_success = sum(1 for r in results if r['failure_reason'] is None)
    failure_counts = defaultdict(int)
    for r in results:
        reason = r['failure_reason']
        if reason is not None:
            # Group exceptions under a common key for the summary table,
            # but keep individual messages for the detailed log
            key = reason if not reason.startswith('exception:') else 'exception'
            failure_counts[key] += 1

    with open(f'{DATA_DIR}/corrections.txt', 'w') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"  Structure fixer — summary ({n_total} input systems)\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"  Successfully written          : {n_success:>6}  ({100*n_success/n_total:.1f}%)\n")
        f.write(f"  Failed                        : {n_total - n_success:>6}  ({100*(n_total - n_success)/n_total:.1f}%)\n\n")

        f.write(f"  --- Failure breakdown ---\n")
        f.write(f"  Null elements in topology     : {failure_counts.get('null_elements_in_topology', 0):>6}\n")
        f.write(f"  Unknown residues (UNK/UNL)     : {failure_counts.get('unknown_residue', 0):>6}\n")
        f.write(f"  Modified residues in shell     : {failure_counts.get('modified_residues_in_shell', 0):>6}\n")
        f.write(f"  Missing atoms in shell         : {failure_counts.get('missing_atoms_in_shell', 0):>6}\n")
        f.write(f"  Branched residues in shell     : {failure_counts.get('branched_residues_in_shell', 0):>6}\n")
        f.write(f"  Uncaught exceptions            : {failure_counts.get('exception', 0):>6}\n\n")

        # Log individual exceptions for debugging
        exceptions = [(r['failure_reason']) for r in results if r['failure_reason'] and r['failure_reason'].startswith('exception:')]
        if exceptions:
            f.write(f"\n  --- Exception details ---\n")
            for exc in exceptions:
                f.write(f"  {exc}\n")

if __name__ == '__main__':
    wrapper(num_cores = 96)
    #main('5yem')
