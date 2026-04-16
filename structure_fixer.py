from src.config import DATA_DIR

# Main structural biology dependencies
import gemmi
from pdbfixer import PDBFixer
from openmm.app import PDBFile

# Basic resources
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
from itertools import product
import string

# File management
import os
import tempfile
import subprocess
from joblib import Parallel, delayed

### Define important variables ###
#---------------------------------

STANDARD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYM', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 
    'TRP', 'TYR', 'VAL', 'HIE', 'HIP', 'HID', 'HSD', 'HSE', 'HSP'}

COMMON_ARTIFACTS = {'02U', '12P', '13P', '144', '15P', '16P', '1EM', '1PE', '1PG', '1PS', '2DP', '2JC', '2NV', '2OP', '2PE', '32M', '33O', '3HR', '3PG', 
                    '3SY', '3V3', '543', '6JZ', '6PE', '7E8', '7E9', '7I7', '7N5', '7PE', '7PG', '7PH', '90A', '9FO', '9JE', '9YU', 'AAE', 'ABA', 'AE3', 
                    'AE4', 'AGA', 'AKR', 'AUC', 'B3H', 'B3P', 'B4T', 'B4X', 'BAM', 'BCN', 'BDN', 'BE7', 'BEN', 'BET', 'BEZ', 'BGL', 'BHG', 'BNG', 'BNZ', 
                    'BOG', 'BTB', 'BU1', 'BXC', 'C10', 'C14', 'C8E', 'CAC', 'CAD', 'CAQ', 'CD4', 'CE1', 'CE9', 'CHT', 'CIT', 'CN3', 'CN6', 'CPS', 'CXE', 
                    'CXS', 'D10', 'D12', 'D1D', 'D22', 'DAO', 'DD9', 'DDQ', 'DDR', 'DEP', 'DET', 'DHB', 'DHJ', 'DIO', 'DKA', 'DMF', 'DMI', 'DMR', 'DOX', 
                    'DPG', 'DR6', 'DRE', 'DTD', 'DTT', 'DTU', 'DTV', 'E4N', 'EAP', 'EEE', 'EPE', 'ETE', 'ETF', 'ETX', 'F09', 'F4R', 'FJO', 'FTT', 'FW5', 
                    'GLV', 'GOL', 'GVT', 'GYF', 'HAE', 'HAI', 'HCA', 'HCS', 'HED', 'HEX', 'HEZ', 'HP6', 'HSG', 'HSH', 'HT3', 'HTG', 'HTH', 'HTO', 'HZA', 
                    'I3C', 'ICT', 'IHP', 'IHS', 'IMD', 'IPH', 'JDJ', 'K12', 'KDO', 'L1P', 'L2C', 'L2P', 'L3P', 'L4P', 'LAC', 'LDA', 'LI1', 'LMR', 'LMT', 
                    'LMU', 'LUT', 'M2M', 'MAC', 'MAE', 'MB3', 'MBN', 'MBO', 'MC3', 'ME2', 'MEG', 'MES', 'MLA', 'MLI', 'MLT', 'MPD', 'MPO', 'MRD', 'MSE', 
                    'MYR', 'N8E', 'NBN', 'NET', 'NEX', 'NHE', 'O4B', 'OCT', 'OES', 'OGA', 'OP2', 'OTE', 'OXM', 'P03', 'P15', 'P1O', 'P22', 'P25', 'P2K', 
                    'P33', 'P3G', 'P4C', 'P4G', 'P4K', 'P6G', 'PA8', 'PC8', 'PD7', 'PE3', 'PE4', 'PE5', 'PE6', 'PE7', 'PE8', 'PEG', 'PEP', 'PEU', 'PEX', 
                    'PG0', 'PG4', 'PG5', 'PG6', 'PG8', 'PGE', 'PGF', 'PGO', 'PGR', 'PHB', 'PHQ', 'PL9', 'PLC', 'PMS', 'PPI', 'PQ9', 'PQE', 'PTD', 'PUT', 
                    'PVO', 'PX2', 'PX4', 'QGT', 'QJE', 'QLB', 'RG1', 'RWB', 'SAR', 'SEP', 'SGM', 'SIN', 'SOG', 'SP5', 'SPD', 'SPJ', 'SPM', 'SPZ', 'SQU', 
                    'SRT', 'TAM', 'TAR', 'TAU', 'TBU', 'TCE', 'TCN', 'TEA', 'TFA', 'THE', 'TLA', 'TMA', 'TOE', 'TPO', 'TRD', 'TRS', 'UMQ', 'UND', 'V1J', 
                    'VX', 'XAT', 'XP4', 'XPA', 'XPE', 'Y69'}

VALID_BOND_ATOMS = {'C', 'N', 'O', 'S', 'P', 'B'}


WATER_NAMES = {"HOH", "WAT", "DOD", "H2O", "TIP3", "SPC", "TIP"}

METALLOCOFACTORS = {'HEM', 'SF4', 'MGD'}

FIXED_RESIDUES = STANDARD_AA | METALLOCOFACTORS | {'LIG', 'HOH', 'WAT', 'TIP3', 'SOL', 'OPC', 'ACE', 'NME'}
CUSTOM_SUBSTITUTIONS = {'SEC': 'CYS', '0A8': 'CYS'}

CLASH_RADIUS = 1.8
CHAIN_RADIUS = 4.0
SHELL_RADIUS = 6.0

### Helper functions ###
#-----------------------

def normalize(s: str) -> str:
    return s.strip('\'"').replace("\\", "")

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

def load_ccd(comp_ids: set[str]) -> dict[str, set[str]]:
    """
    Parse components.cif and extract heavy atom names for
    only the residue types we actually need.
    """

    doc = gemmi.cif.read(f'{DATA_DIR}/mmCIF/ccd/components.cif')
    expected = {}

    for block in doc:
        if block.name not in comp_ids:
            continue

        table = block.find(
            "_chem_comp_atom.",
            ["atom_id", "type_symbol"],
        )
        if not table:
            continue

        heavy = set()
        for row in table:
            if row[1] not in ("H", "D") and row[0] != "OXT":
                heavy.add(row[0])

        expected[block.name] = heavy

        # Stop early if we've found everything
        if len(expected) == len(comp_ids):
            break

    return expected

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
	if {'C', 'N'}.issubset(elements) or {'C', 'O'}.issubset(elements):
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

### Core functions ###
#----------------

def remove_artifacts_and_fix_quotes(pdb_id, tmp_dir):
    """
    Remove common artifacts and handle file formatting issues

    Parameters
    ----------
    pdb_id [str]
    tmp_dir [tempfile.TemporaryDirectory()]
    """

    out_list = []

    with open(f'{DATA_DIR}/mmCIF/raw/{pdb_id}.cif', 'r') as infile:
        buffer = "" # Buffer for handling multi-line entries with unbalanced quotes
        for line in infile:
            line = line.strip()
                
            if not line.startswith(('HETATM', 'ATOM')):
                stripped_line = line.strip()
                if buffer:
                    buffer += ' ' + stripped_line
                    if buffer.count('"') % 2 == 0:
                        line = buffer
                        buffer = ""
                    else:
                        continue
                else:
                    if stripped_line.count('"') % 2 != 0:
                        buffer = stripped_line
                        continue
                    else:
                        line = stripped_line

            # Handle ATOM/HETATM lines
            else:
                columns = line.split()

                if len(columns) < 4:
                    continue
                if columns[5] in COMMON_ARTIFACTS:
                    continue

                columns[18] = columns[6]
                line = ' '.join(columns)

            out_list.append(line)

    with open(f'{tmp_dir}/{pdb_id}.cif', 'w') as outfile:
        outfile.write('\n'.join(out_list))

def select_conformer(pdb_id, tmp_dir):
    """
    Reduce residues to maximum atoms / maximum occupancy conformers.

    Parameters
    ----------
    pdb_id [str]
    tmp_dir [tempfile.TemporaryDirectory()]
    """

    structure = gemmi.read_structure(f'{tmp_dir}/{pdb_id}.cif')

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
    
    def resolve_overlaps(self, structure: gemmi.Structure, overlaps: List[str]):
        """
        Keep the chain with the most heavy atoms
        """

        model = structure[0]

        for overlap_pair in overlaps:
            chain1_id, chain2_id = overlap_pair.split(",")

            if _has_chain(model, chain1_id) and _has_chain(model, chain2_id):
                chain1_count = _count_heavy_atoms(model, chain1_id)
                chain2_count = _count_heavy_atoms(model, chain2_id)

                if chain1_count > chain2_count:
                    _remove_chain(model, chain2_id)
                else:
                    _remove_chain(model, chain1_id)

    def merge_bonded_chains(self, structure: gemmi.Structure, bonds: List[str]):
        """
        Merge chains and rename to alphabetically first chain
        """

        model = structure[0]

        # Build connected components
        chain_sets = [set(bond.split(",")) for bond in bonds]
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

        chain_lists = [sorted(list(x)) for x in chain_sets]

        # Merge chain groups
        for chain_list in chain_lists:
            new_chain_name = chain_list[0]

            all_residues = []          # list of (gemmi.Atom, seqid_tuple)
            residue_counts = []
            chains_to_remove = []

            for chain_id in chain_list:
                if not _has_chain(model, chain_id):
                    continue

                chain = model[chain_id]
                res_count = 0
                for residue in chain:
                    res_count += 1
                    all_residues.append((residue.clone(), residue.seqid.num))

                residue_counts.append(res_count)
                chains_to_remove.append(chain_id)

            if not all_residues:
                return
            
            # Remove old chains
            for chain_id in chains_to_remove:
                _remove_chain(model, chain_id)

            # Build merged chain with preserved residues
            new_chain = gemmi.Chain(new_chain_name)
            for new_index, (residue, _) in enumerate(all_residues, start=1):
                residue.seqid = gemmi.SeqId(str(new_index))
                new_chain.add_residue(residue)

            model.add_chain(new_chain)

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

    model = structure[0]
    chain = model[chain_id]
    ccd_codes = set([res.name for res in chain])

    # Load official CCD depositions
    expected_atoms = load_ccd(ccd_codes)

    # Compare observed vs expected
    for residue in chain:
        expected = expected_atoms.get(residue.name)

        if expected is None:
            print(f'Error: CCD code {residue.name} not found in components.cif')

        observed = {atom.name for atom in residue if not atom.element.name in {'H', 'D'}}

        # Weird fix: remove backslashes from expected
        expected = {normalize(s) for s in expected}
        observed = {normalize(s) for s in observed}

        absent = expected - observed
        absent = [x[0] for x in absent]

        if len(absent) > 1 or absent[0] != 'O':
            return False

    return True

def build_pdb(structure: gemmi.Structure, chain_id):
    """
    1. Applies symmetry expansion
    2. Selects only neighboring chains
    3. Renames chains and updates SEQRES
    4. Writes as PDB
    """

    model = structure[0]
    cell = structure.cell
    sg = structure.find_spacegroup()
    target_chain = model[chain_id]

    # Create KDTree for fast contact checks
    target_coords = []
    for res in target_chain:
        for atom in res:
            if not atom.element.name in {'H', 'D'}:
                target_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])

    target_tree = KDTree(np.array(target_coords))

    # Symmetry expansion and neighbor selection
    neighbors = []  # list of (transformed_chain, source_chain_name)

    for chain in model:
        if sg is not None:
            for op in sg.operations():
                # Create 3x3x3 unit cell box
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
                    neighbors.append((transformed, chain.name))

    # Build output structure
    new_structure = gemmi.Structure()
    new_structure.cell = cell
    new_structure.spacegroup_hm = 'P 1'

    new_model = gemmi.Model(0)

    # Target chain → Z
    target_clone = target_chain.clone()
    target_clone.name = "Z"
    new_model.add_chain(target_clone)

    # Neighbors → A, B, C, ...
    available = list("ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxyz0123456789")
    source_map = {"Z": chain_id}  # new_name -> source_chain_name

    for i, (nchain, src_name) in enumerate(neighbors):
        new_name = available[i]
        nchain.name = new_name
        new_model.add_chain(nchain)
        source_map[new_name] = src_name

    new_structure.add_model(new_model)

    # ── Rebuild entities for SEQRES ──

    # Build lookup: source chain name -> Entity in original structure
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

    # Assign subchains so entities connect to chains during PDB writing
    _assign_subchain_ids(new_structure)

    return new_structure

def fix_missing_and_nonstandard_residues(basename: str, ligand_coords: np.array, tmp_dir):
    """
    Use PDBFixer to find nonstandard residues, missing residues and unresolved atoms.

    If **all** nonstandard residues are farther than *distance_cutoff* Å from
	every atom in *ligand_coords*, they are silently replaced by their standard
	counterparts and the PDB is overwritten in-place.

	If any nonstandard residue has at least one atom within *distance_cutoff* Å
	of a ligand atom, the file is left untouched and the function returns
	``False`` so the caller can abort.

    Also add capping
    """

    fixer = PDBFixer(filename = f'{DATA_DIR}/pdb/raw/{basename}.pdb')
    fixer.findNonstandardResidues()

    # Build a list of residues per chain for neighbor lookup
    chain_residues = {}
    chain_id_map = {}  # residue index -> chain id
    for chain in fixer.topology.chains():
        chain_residues[chain.index] = list(chain.residues())
        for residue in chain.residues():
            chain_id_map[residue.index] = chain.id

    # Apply custom substitutions and default-to-ALA fallback
    already_flagged = {r for r, _ in fixer.nonstandardResidues}

    # Remove any chain Z residues that PDBFixer auto-flagged
    fixer.nonstandardResidues = [
        (r, sub) for r, sub in fixer.nonstandardResidues
        if chain_id_map.get(r.index) != "Z"
    ]  

    for residue in fixer.topology.residues():
        if chain_id_map.get(residue.index) == "Z":
            continue

        elements = {atom.element for atom in residue.atoms()}
        if None in elements:
            return

        if residue.name not in FIXED_RESIDUES and residue not in already_flagged:
            if is_nonstandard_residue(residue, chain_residues):
                replacement = CUSTOM_SUBSTITUTIONS.get(residue.name, "ALA")
                fixer.nonstandardResidues.append((residue, replacement))

    # Check each nonstandard residue for proximity to any ligand atom
    if fixer.nonstandardResidues:
        for residue, _replacement in fixer.nonstandardResidues:
            for atom in residue.atoms():
                pos = fixer.positions[atom.index]
                atom_coord = np.array([pos.x * 10.0, pos.y * 10.0, pos.z * 10.0])
                dists = np.linalg.norm(ligand_coords - atom_coord, axis=1)
                if dists.min() < SHELL_RADIUS:
                    return 'modified_in_shell'
        # All nonstandard residues are far from ligands – safe to replace
        fixer.replaceNonstandardResidues()

    fixer.findMissingResidues()
    fixer.findMissingAtoms()

    # Exclude chain Z from missing residues
    fixer.missingResidues = {
        key: val for key, val in fixer.missingResidues.items()
        if chain_id_map.get(key[0]) != "Z"
        # key is (chain_index, residue_index) — but safer to check via chain directly
    }

    # Exclude chain Z from missing atoms
    fixer.missingAtoms = {
        residue: atoms for residue, atoms in fixer.missingAtoms.items()
        if chain_id_map.get(residue.index) != "Z"
    }

    # Check if any remaining residue with missing atoms is near the ligand
    for residue in list(fixer.missingAtoms.keys()):
        for atom in residue.atoms():
            pos = fixer.positions[atom.index]
            atom_coord = np.array([pos.x * 10.0, pos.y * 10.0, pos.z * 10.0])
            dists = np.linalg.norm(ligand_coords - atom_coord, axis=1)
            if dists.min() < SHELL_RADIUS:
                return 'missing_atoms_in_shell'

    fixer.addMissingAtoms()

    with open(f'{DATA_DIR}/pdb/fixed/{basename}.pdb', "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    return 'ok'

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
            'failure_reason': None,  # None means success
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Step 1: remove artifacts and fix file formatting
            remove_artifacts_and_fix_quotes(pdb_id, tmp_dir)

            # Step 2: reduce conformer
            structure = select_conformer(pdb_id, tmp_dir)

            # Step 3: resolve steric clashes and missing bonds between chains
            overlap_resolver = OverlapResolver()
            contact_info, bonds_to_add, overlaps_to_resolve = overlap_resolver.detect_contacts(structure)
            overlap_resolver.resolve_overlaps(structure, overlaps_to_resolve)
            overlap_resolver.merge_bonded_chains(structure, bonds_to_add)

            # 3.2: save cleaned mmCIF structure
            structure.make_mmcif_document().write_file(f'{DATA_DIR}/mmCIF/clean/{pdb_id}.cif')

            # Step 4: select possible ligand chains
            ligand_chains = select_ligand_chains(structure)

            # Step 5: loop over ligand chains
            for chain_id in ligand_chains:

                # 5.1: Any unresolved ligand atoms?
                fully_resolved = detect_unresolved_ligand_atoms(structure, chain_id)
                if fully_resolved:

                    # 5.2: convert to PDB
                    ccd_codes = [res.name for res in structure[0][chain_id]]
                    lig_name = '-'.join(ccd_codes)

                    new_structure = build_pdb(structure, chain_id)
                    basename = f'{pdb_id}_{chain_id}'
                    new_structure.write_pdb(f'{DATA_DIR}/pdb/raw/{basename}.pdb')

                    # 5.3 missing and nonstand residues with PDBFixer
                    update_element_positions(f'{DATA_DIR}/pdb/raw/{basename}.pdb')
                    ligand_coords = _get_chain_coords(new_structure, 'Z')
                    fixer_status = fix_missing_and_nonstandard_residues(basename, ligand_coords, tmp_dir)
                    if fixer_status == 'modified_in_shell':
                        flags['has_modified_residues_in_shell'] = True
                        flags['failure_reason'] = 'modified_residues_in_shell'
                    elif fixer_status == 'missing_atoms_in_shell':
                        flags['has_missing_atoms_in_shell'] = True
                        flags['failure_reason'] = 'missing_atoms_in_shell'
                    elif fixer_status is None:
                        flags['failure_reason'] = 'null_elements_in_topology'

    except Exception as e:
        print(f'{pdb_id} - {e}')
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
        f.write(f"  Modified residues in shell     : {failure_counts.get('modified_residues_in_shell', 0):>6}\n")
        f.write(f"  Missing atoms in shell         : {failure_counts.get('missing_atoms_in_shell', 0):>6}\n")
        f.write(f"  Uncaught exceptions            : {failure_counts.get('exception', 0):>6}\n\n")

        # Log individual exceptions for debugging
        exceptions = [(r['failure_reason']) for r in results if r['failure_reason'] and r['failure_reason'].startswith('exception:')]
        if exceptions:
            f.write(f"\n  --- Exception details ---\n")
            for exc in exceptions:
                f.write(f"  {exc}\n")

if __name__ == '__main__':
    wrapper(num_cores = 32)
