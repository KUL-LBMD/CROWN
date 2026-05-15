from src.config import DATA_DIR
 
import os
import shutil
import tempfile
import numpy as np
import string
 
from scipy.spatial import KDTree
from pdbfixer import PDBFixer
from openmmforcefields.generators import SystemGenerator
from openmm.app import PDBFile, Modeller, Topology, Simulation, CutoffNonPeriodic
from openff.toolkit import Molecule
from openff.units import unit as openff_unit
from openff.nagl_models import list_available_nagl_models
from openff.nagl import GNNModel
from openmm import CustomExternalForce, LangevinMiddleIntegrator, unit, Platform
import logging
from joblib import Parallel, delayed
 
from concurrent.futures import TimeoutError as FuturesTimeoutError
import functools
import xml.etree.ElementTree as ET
 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
 
# Restrict sqm/antechamber to 1 thread per worker
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ["OPENMM_DEFAULT_PLATFORM"] = "CPU"
os.environ["OPENMM_CPU_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
 
# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================
MOBILE_RADIUS = 0.6  # Distance in nm (6 Å = 0.6 nm)
FIX_STRENGTH = 1000000.0  # kJ/mol/nm² - very high to effectively freeze atoms
TETHER_STRENGTH = 10 # kcal/(mol*A^2). Default parameter in MOE
TETHER_FLATBOTTOM = 0.25 # Within 0.25 A radius, atoms feel no tethering force
MINIMIZATION_STEPS = 5000
ENERGY_REPORT_INTERVAL = 50
TEMPERATURE = 300  # Kelvin
TIMESTEP = 0.002  # picoseconds
PH = 7.4
NONBONDED_CUTOFF = 0.8 # Cutoff for interactions
 
FORCEFIELD_LIST = [
	'amber19/protein.ff19SB.xml',
	'amber19/DNA.OL21.xml',
	'amber14/RNA.OL3.xml',
	'amber19/opc3.xml',
	f'{DATA_DIR}/custom_xml/forcefield/HEM.xml',
	f'{DATA_DIR}/custom_xml/forcefield/MGD.xml',
	f'{DATA_DIR}/custom_xml/forcefield/SF4.xml',
]
 
STANDARD_AMINO_ACIDS = {
	'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYM', 'GLN', 'GLU', 'GLY',
	'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR',
	'TRP', 'TYR', 'VAL', 'HIE', 'HIP', 'HID', 'HSD', 'HSE', 'HSP',
	'ACE', 'NME'
}
 
STANDARD_BASES = {'A', 'U', 'G', 'C', 'DA', 'DT', 'DG', 'DC',
	'A3', 'A5', 'U3', 'U5', 'G3', 'G5', 'C3', 'C5', 'DA3', 'DA5', 'DT3', 'DT5', 'DG3', 'DG5', 'DC3', 'DC5'
}
 
WATER_NAMES = {'HOH', 'WAT', 'TIP3', 'DOD', 'O'}
METALLOCOFACTORS = {'HEM', 'SF4', 'MGD'}
TEMPLATES_TO_REMOVE = {'AG1', 'Ce', 'Cr', 'CU1', 'EU3', 'FE2', 'TL1', 'Sm'}

# =====================================================
# Helper functions
# =====================================================

def timeout(seconds=300):
    """Cross-platform timeout decorator"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except FuturesTimeoutError:
                    raise TimeoutError(f"Function exceeded {seconds}s timeout")
        return wrapper
    return decorator

def _get_file_length(path):

	with open(path, 'r') as f:
		lines = [line.strip() for line in f]
		return len(lines)

def _clean_ff(ff):
	"""
	Remove non-identical matching templates.
	"""

	for key in TEMPLATES_TO_REMOVE:
		del ff._templates[key]

	# Remove from signature matching as well
	for sig, templates in list(ff._templateSignatures.items()):
		ff._templateSignatures[sig] = [t for t in templates if not t.name in TEMPLATES_TO_REMOVE]

	return ff

def _clean_file(pdb_path):
	lines_to_keep = []
	with open(pdb_path, 'r') as f:
		for line in f:
			line = line.strip()
			parts = line.split()

			if line.startswith(('ATOM', 'HETATM')):
				res_name = line[17:20].strip()
				if res_name in WATER_NAMES:
					line = line[:17] + 'HOH' + line[20:]

			if parts[0] not in {'HET', 'CRYST1'}:
				lines_to_keep.append(line)

	with open(pdb_path, 'w') as f:
		f.write('\n'.join(lines_to_keep))

def _rename_single_atom_residues(pdb_path):
    pdb = PDBFile(pdb_path)
    modified = False

    modeller = Modeller(pdb.topology, pdb.positions)

    atoms_to_remove = []
    for atom in modeller.topology.atoms():
        if atom.element.symbol in {'H', 'D'}:
            atoms_to_remove.append(atom)

    if atoms_to_remove:
        modeller.delete(atoms_to_remove)
        modified = True

    if modified:
        with open(pdb_path, 'w') as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f)

def _assign_charges_with_fallback(molecule: Molecule) -> Molecule:
    """
    Assign AM1-BCC partial charges to an OpenFF Molecule, with NAGL as fallback.

    AM1-BCC via AmberTools (sqm) can fail for large or highly-charged molecules
    (e.g. NADP+, CoA). NAGL is a GNN-based charge model that is much more robust
    and is already available in this environment.

    Parameters
    ----------
    molecule : openff.toolkit.Molecule
        The molecule to charge. Modified in-place and returned.

    Returns
    -------
    openff.toolkit.Molecule
        The same molecule with partial_charges populated.
    """

    # --- Attempt 1: standard am1bcc via AmberTools (sqm) ---
    try:
        molecule.assign_partial_charges('gasteiger')
        logger.info(f"Charges assigned via am1bcc for '{molecule.name}'.")
        return molecule
    except Exception as e:
        logger.warning(
            f"am1bcc charge assignment failed for '{molecule.name}' "
            f"(net charge {molecule.total_charge}): {e}\n"
            f"Falling back to NAGL."
        )

    # --- Attempt 2: NAGL GNN charge model ---
    try:
        # Prefer the most recent am1bcc-equivalent model
        available_models = list_available_nagl_models()

        # Prefer the stable release am1bcc model
        preferred = [m for m in available_models if 'am1bcc' in str(m)]
        model_path = preferred[-1] if preferred else available_models[-1]
        logger.info(f"Using NAGL model: {model_path}")

        # NAGLToolkitWrapper must be called directly; nagl_model is not a valid
        # kwarg to the standard assign_partial_charges() dispatch method
        nagl_model = GNNModel.load(model_path, eval_mode=True)
        charges = nagl_model.compute_property(molecule, as_numpy=True)
        molecule.partial_charges = charges * openff_unit.elementary_charge

        logger.info(f"Charges assigned via NAGL for '{molecule.name}'.")
        return molecule

    except Exception as e:
        logger.warning(f"NAGL charge assignment also failed for '{molecule.name}': {e}")

    try:
        molecule.assign_partial_charges('gasteiger')
        return molecule
    except Exception as e:
        raise

def _find_cofactors(pdb_path):
	"""
	Find all metallocofactor entries in a pdb file
	"""

	amber_residues = set()

	with open(pdb_path, 'r') as f:
		for line in f:
			if line.startswith(('ATOM', 'HETATM')):
				resname = line[17:20].strip()
				if resname in METALLOCOFACTORS:
					amber_residues.add(resname)

	return amber_residues

def _load_bond_templates(forcefield_paths, resname_filter=None):
    """
    Parse OpenMM-format forcefield XMLs and extract intra-residue bond
    definitions keyed by residue name.
 
    Only paths that resolve to an actual file on disk are parsed - bare
    Amber/CHARMM names like 'amber19/protein.ff19SB.xml' are resolved
    internally by OpenMM's data path and are skipped here.
 
    Parameters
    ----------
    forcefield_paths : iterable of str
        Candidate XML paths (typically FORCEFIELD_LIST).
    resname_filter : set[str] or None
        If given, only residues whose name is in this set are loaded.
 
    Returns
    -------
    dict[str, list[tuple[str, str]]]
        {resname: [(atom_name_a, atom_name_b), ...]}
    """
    templates = {}

    for path in forcefield_paths:
        if not os.path.isfile(path):
            continue
        try:
            tree = ET.parse(path)
        except ET.ParseError as e:
            logger.warning(f"Could not parse forcefield XML '{path}': {e}")
            continue
 
        root = tree.getroot()
        for residue in root.findall('.//Residues/Residue'):
            resname = residue.get('name')
            if resname is None:
                continue
            if resname_filter is not None and resname not in resname_filter:
                continue
 
            bonds = []
            for bond in residue.findall('Bond'):
                # OpenMM forcefield XMLs use atomName1/atomName2;
                # residue-definition XMLs use from/to. Accept either.
                a = bond.get('atomName1') or bond.get('from')
                b = bond.get('atomName2') or bond.get('to')
                if a and b:
                    bonds.append((a, b))
 
            if bonds:
                templates.setdefault(resname, []).extend(bonds)
 
    return templates
 
def _add_bonds(topology, resname_set, bond_templates):
    """
    Add missing intra-residue bonds for the requested residues, using bond
    definitions parsed from forcefield XML templates (matched by atom name).
 
    Parameters
    ----------
    topology : openmm.app.Topology
    resname_set : set[str]
        Residue names to process (e.g. METALLOCOFACTORS).
    bond_templates : dict[str, list[tuple[str, str]]]
        Output of _load_bond_templates().
    """
    # Symmetric lookup of existing bonds
    existing = {frozenset((b[0].index, b[1].index)) for b in topology.bonds()}
 
    n_added = 0
    n_missing_atom = 0
    for residue in topology.residues():
        if residue.name not in resname_set:
            continue
        if residue.name not in bond_templates:
            logger.warning(
                f"No bond template found for residue '{residue.name}' in any "
                f"forcefield XML; skipping bond reconstruction for this residue."
            )
            continue
 
        atoms_by_name = {atom.name: atom for atom in residue.atoms()}
        for a_name, b_name in bond_templates[residue.name]:
            a = atoms_by_name.get(a_name)
            b = atoms_by_name.get(b_name)
            if a is None or b is None:
                n_missing_atom += 1
                continue
            key = frozenset((a.index, b.index))
            if key in existing:
                continue
            topology.addBond(a, b)
            existing.add(key)
            n_added += 1
 
    if n_missing_atom:
        logger.warning(
            f"_add_bonds: {n_missing_atom} bond definition(s) skipped "
            f"because the atom name was not found in the topology"
        )
    logger.info(f"_add_bonds: added {n_added} bond(s) from forcefield templates")

def _add_seqres_with_caps(input_pdb: str, output_pdb: str):
	"""
	Add SEQRES records with ACE/NME caps to a PDB file.
	PDBFixer will then detect these as 'missing' and build them.
	"""

	# First, read the structure to get chain sequences
	pdb = PDBFile(input_pdb)

	# Build sequence for each chain
	chain_sequences = {}
	for chain in pdb.topology.chains():
		residues = list(chain.residues())
		protein_residues = [r for r in residues if r.name in STANDARD_AMINO_ACIDS]

		if protein_residues:
			seq = [r.name for r in protein_residues]
			# Add ACE at start, NME at end
			if protein_residues[0].name != 'ACE':
				seq = ['ACE'] + seq
			if protein_residues[-1].name != 'NME':
				seq = seq + ['NME']
			chain_sequences[chain.id] = seq

	# Now write modified PDB with SEQRES records
	with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
		# First write SEQRES records for each chain
		for chain_id, seq in chain_sequences.items():
			# SEQRES records: max 13 residues per line
			for i in range(0, len(seq), 13):
				chunk = seq[i:i+13]
				line_num = (i // 13) + 1
				seqres_line = f"SEQRES {line_num:>3} {chain_id} {len(seq):>4}  "
				seqres_line += " ".join(f"{res:>3}" for res in chunk)
				f_out.write(seqres_line + '\n')

		# Then copy the rest of the file (skip existing SEQRES lines)
		for line in f_in:
			if not line.startswith('SEQRES'):
				f_out.write(line)

def _cap_dna_termini(input_pdb: str, output_pdb: str):
    """
    Rename DNA terminal residues to match Amber OL21 templates and strip
    the 5'-terminal phosphate so it matches the 5'-OH form of DX5.

    - First DNA residue of each chain: DX -> DX5  (5'-OH)
    - Last DNA residue of each chain:  DX -> DX3  (3'-OH)
    - Single-residue DNA chain:        DX -> DXN  (both ends capped)

    PDBFixer.findMissingAtoms()/addMissingHydrogens() will then add HO5'
    and HO3' to satisfy the renamed templates.
    """
    pdb = PDBFile(input_pdb)
    modeller = Modeller(pdb.topology, pdb.positions)

    atoms_to_remove = []

    for chain in modeller.topology.chains():
        dna_residues = [r for r in chain.residues() if r.name in STANDARD_BASES]
        if not dna_residues:
            continue

        if len(dna_residues) == 1:
            # Single-nucleotide chain: both termini on one residue
            only = dna_residues[0]
            only.name = only.name + 'N'
            for atom in only.atoms():
                if atom.name in {'P', 'OP1', 'OP2', 'OP3', 'O1P', 'O2P', 'O3P'}:
                    atoms_to_remove.append(atom)
            continue

        # 5' terminus: rename + strip dangling phosphate
        first = dna_residues[0]
        first.name = first.name + '5'
        for atom in first.atoms():
            if atom.name in {'P', 'OP1', 'OP2', 'OP3', 'O1P', 'O2P', 'O3P'}:
                atoms_to_remove.append(atom)

        # 3' terminus: No action needed
        last = dna_residues[-1]
        last.name = last.name + '3'

    if atoms_to_remove:
        modeller.delete(atoms_to_remove)

    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

def _prepare_amber(tmp_dir, pdb_path, special_residues):
	"""
	Prepare modeller and force field list for special AMBER residues
	"""

	basename = pdb_path.split('/')[-1][:-4]

	# DNA terminal renaming + 5'-phosphate stripping
	capped_path = f'{tmp_dir}/{basename}_dnacap.pdb'
	_cap_dna_termini(pdb_path, capped_path)

	# Add SEQRES with caps
	seqres_path = f'{tmp_dir}/{basename}_seqres.pdb'
	_add_seqres_with_caps(capped_path, seqres_path)

	Modeller.loadHydrogenDefinitions(f'{DATA_DIR}/custom_xml/protonation/special_residues_amber.xml')

	fixer = PDBFixer(capped_path)
	fixer.findMissingResidues()
	fixer.findMissingAtoms()
	fixer.addMissingAtoms()
	fixer.addMissingHydrogens(PH)

	logging.getLogger("openff").setLevel(logging.ERROR)

	if special_residues:
		Modeller.loadHydrogenDefinitions(f'{DATA_DIR}/custom_xml/protonation/special_residues_amber.xml')
		modeller = Modeller(fixer.topology, fixer.positions)
		modeller.addHydrogens(pH=PH)
		bond_templates = _load_bond_templates(FORCEFIELD_LIST, resname_filter=special_residues)
		_add_bonds(modeller.topology, special_residues, bond_templates)
	else:
		modeller = Modeller(fixer.topology, fixer.positions)

	return modeller

def _get_rebuilt_atom_indices(original_pdb_path, topology, positions, tol_nm=0.005):
	"""
	Identify atoms that were rebuilt by PDBFixer.

	An atom is considered 'rebuilt' if it is not present at the same
	coordinates in the original PDB. ACE/NME caps and any other residues without
	a CA atom in the new topology are also treated as rebuilt.

	Parameters
	----------
	original_pdb_path : str
		Path to the un-fixed PDB (coordinates in Angstroms).
	topology : openmm.app.Topology
		Current (post-PDBFixer) topology.
	positions : list of openmm.Vec3 with units
		Current positions matching `topology`.
    tol_nm : float
        Max distance (nm) for matching a CA against the original.
        Default 0.05 nm = 0.5 Å, which is generous given PDBFixer doesn't
        perturb existing atoms.

    Returns
    -------
    set[int]
        Atom indices in `topology` belonging to rebuilt residues.
    """
    
	original_coords = []
	with open(original_pdb_path, 'r') as f:
		for line in f:
			if line.startswith(('HETATM', 'ATOM')):
				x = float(line[30:38]) * 0.1
				y = float(line[38:46]) * 0.1
				z = float(line[46:54]) * 0.1
				original_coords.append((x,y,z))

	if not original_coords:
		return set()
	
	tree = KDTree(np.asarray(original_coords))
	rebuilt_atom_indices = set()
	for residue in topology.residues():
		# ACE/NME caps and similar have no CA — always rebuilt
		if residue.name in {'ACE', 'NME'}:
			rebuilt_atom_indices.update(a.index for a in residue.atoms())
			continue

		# Only protein residues are candidates for CA-based matching
		if residue.name not in STANDARD_AMINO_ACIDS:
			continue

		for atom in residue.atoms():
			pos = np.asarray(positions[atom.index].value_in_unit(unit.nanometer))
			dist, _ = tree.query(pos, k=1)
			if dist > tol_nm:
				rebuilt_atom_indices.add(atom.index)

	return rebuilt_atom_indices

def _strip_distant_waters(modeller, ligand_molecules, cutoff_nm=0.4):
    """
    Delete waters whose oxygen is further than `cutoff_nm` from any ligand
    or metallocofactor heavy atom. Operates on `modeller` in place.
 
    Reference set:
      * heavy atoms of every Molecule in `ligand_molecules` (from .sdf inputs)
      * heavy atoms of every residue in the modeller topology whose name is in
        METALLOCOFACTORS (HEM, MGD, SF4) - these enter via the receptor PDB
        rather than as separate SDFs and would otherwise be invisible here.
 
    Parameters
    ----------
    modeller : openmm.app.Modeller
    ligand_molecules : iterable of openff.toolkit.Molecule
        Ligands with at least one conformer set; positions read from
        conformers[0] (in Å).
    cutoff_nm : float
        Distance cutoff in nm. Default 0.4 nm = 4 Å.
 
    Returns
    -------
    int
        Number of water residues removed.
    """
    # Modeller positions in nm (used for both cofactor refs and water queries)
    positions = np.array(
        [p.value_in_unit(unit.nanometer) for p in modeller.positions]
    )
 
    # Collect reference heavy-atom positions in nm
    ref_positions = []
 
    # (1) SDF-derived ligands
    for mol in ligand_molecules:
        coords_nm = mol.conformers[0].to(openff_unit.nanometer).magnitude
        for atom, xyz in zip(mol.atoms, coords_nm):
            if atom.atomic_number != 1:
                ref_positions.append(xyz)

    # (2) Metallocofactors already in the receptor topology
    n_cofactor_atoms = 0
    for residue in modeller.topology.residues():
        if residue.name in STANDARD_AMINO_ACIDS or residue.name in WATER_NAMES:
            continue
        for atom in residue.atoms():
            if atom.element.symbol != 'H':
                ref_positions.append(positions[atom.index])
                n_cofactor_atoms += 1
    if n_cofactor_atoms:
        logger.info(
            f"_strip_distant_waters: seeded KDTree with {n_cofactor_atoms} "
            f"metallocofactor heavy atom(s) in addition to SDF ligand atoms"
        )
 
    if not ref_positions:
        return 0
 
    tree = KDTree(np.asarray(ref_positions))
 
    waters_to_remove = []
    for residue in modeller.topology.residues():
        if residue.name not in WATER_NAMES:
            continue
        oxygen = next(
            (a for a in residue.atoms() if a.element.symbol == 'O'), None
        )
        if oxygen is None:
            continue
        d, _ = tree.query(positions[oxygen.index], k=1)

        if d > cutoff_nm:
            waters_to_remove.append(residue)

    if waters_to_remove:
        modeller.delete(waters_to_remove)
    return len(waters_to_remove)

# ===================================================
# Stage helpers
# ===================================================

def _add_ligands_to_modeller(modeller, input_dir):
	"""
	Read all .sdf ligands from the input dir, charge them, strip waters that
	are >4 Å from any ligand heavy atom, and add the ligands to the modeller.
 
	Returns
	-------
	(ligand_entries, ligand_molecules)
		ligand_entries  : list of (basename, Molecule, [atom_index, ...])
		ligand_molecules: list of Molecule (in the same order)
	"""
	# Pass 1: read + charge (don't add to modeller yet, so water-stripping
	# can happen against the protein-only frame and indices stay stable)
	pending_ligands = []
	for ligand_file in sorted(os.listdir(f'{DATA_DIR}/systems/{input_dir}')):
		if not ligand_file.endswith('.sdf'):
			continue
		basename = ligand_file.replace('.sdf', '')
		ligand_mol = Molecule.from_file(
			f'{DATA_DIR}/systems/{input_dir}/{ligand_file}',
			allow_undefined_stereo=True,
		)
		ligand_mol = _assign_charges_with_fallback(ligand_mol)
		pending_ligands.append((basename, ligand_mol))
 
	n_removed = _strip_distant_waters(
		modeller, [m for _, m in pending_ligands], cutoff_nm=0.4,
	)
	logger.info(f"Stripped {n_removed} waters > 4 Å from any ligand atom")
 
	# Pass 2: add ligands; record the index ranges
	ligand_entries = []
	ligand_molecules = []
	for basename, ligand_mol in pending_ligands:
		ligand_molecules.append(ligand_mol)
		ligand_topology = ligand_mol.to_topology().to_openmm()
		ligand_positions = ligand_mol.conformers[0].to_openmm()
		offset = modeller.topology.getNumAtoms()
		n_atoms = ligand_topology.getNumAtoms()
		modeller.add(ligand_topology, ligand_positions)
		ligand_entries.append(
			(basename, ligand_mol, list(range(offset, offset + n_atoms)))
		)
 
	return ligand_entries, ligand_molecules

def _collect_ligand_and_cofactor_indices(modeller, ligand_entries, all_atoms):
	"""Heavy-atom indices for ligands + metallocofactors (KDTree seed)."""
	indices = set()
	for _, _, atom_idxs in ligand_entries:
		indices.update(i for i in atom_idxs if all_atoms[i].element.symbol != 'H')
	for residue in modeller.topology.residues():
		if residue.name in METALLOCOFACTORS:
			for atom in residue.atoms():
				if atom.element.symbol != 'H':
					indices.add(atom.index)
	return indices 
 
def _compute_mobile_atoms(modeller, ligand_indices, all_atoms):
	"""Heavy atoms within MOBILE_RADIUS of any ligand/cofactor heavy atom."""
	positions = modeller.positions
	all_positions_nm = np.array(
		[p.value_in_unit(unit.nanometer) for p in positions]
	)
	ligand_positions_nm = all_positions_nm[sorted(ligand_indices)]
	tree = KDTree(ligand_positions_nm)
	distances, _ = tree.query(all_positions_nm, k=1)
	mobile_mask = distances <= MOBILE_RADIUS
	return {
		i for i in np.where(mobile_mask)[0].tolist()
		if all_atoms[i].element.symbol != 'H'
	}

def _build_restraints():
	"""
	Construct the two CustomExternalForce objects used to restrain the system.
	"""
 
	nonmobile_restraint = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
	nonmobile_restraint.addGlobalParameter(
		'k', FIX_STRENGTH * unit.kilojoules_per_mole / unit.nanometer**2
	)
	nonmobile_restraint.addPerParticleParameter("x0")
	nonmobile_restraint.addPerParticleParameter("y0")
	nonmobile_restraint.addPerParticleParameter("z0")
 
	# Continuously differentiable flat-bottom tether (smoothstep on [u, d+u]).
	# Energy/force/2nd-derivative are 0 at r=u; energy is linear past r=d+u.
	mobile_restraint = CustomExternalForce(
		'w*('
		'step(r-u)*(1-step(r-(d+u)))*(a*(r-u)^5+b*(r-u)^4+c*(r-u)^3)'  # [u, d+u]
		'+step(r-(d+u))*d*(r-u)'                                       # [d+u, +inf]
		'); '
		'r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2+eps)'
	)
	mobile_restraint.addGlobalParameter(
		'w', TETHER_STRENGTH * unit.kilocalories_per_mole / unit.angstrom**2
	)
	mobile_restraint.addGlobalParameter('u', TETHER_FLATBOTTOM * unit.angstrom)
	mobile_restraint.addGlobalParameter('a', 3 * unit.angstrom**(-3))
	mobile_restraint.addGlobalParameter('b', -8 * unit.angstrom**(-2))
	mobile_restraint.addGlobalParameter('c', 6 * unit.angstrom**(-1))
	mobile_restraint.addGlobalParameter('d', 1.0 * unit.angstrom)
	mobile_restraint.addGlobalParameter('eps', 1e-16 * unit.nanometer**2)
	mobile_restraint.addPerParticleParameter("x0")
	mobile_restraint.addPerParticleParameter("y0")
	mobile_restraint.addPerParticleParameter("z0")
 
	return nonmobile_restraint, mobile_restraint

def _populate_restraints(modeller, mobile_atoms, skip_indices,
                         nonmobile_restraint, mobile_restraint):
	"""Assign each non-skipped heavy atom to mobile or non-mobile restraint."""
	positions = modeller.positions
	for atom in modeller.topology.atoms():
		if atom.element.symbol == 'H':
			continue
		if atom.index in skip_indices:
			continue

		pos = positions[atom.index].value_in_unit(unit.nanometers)
		if atom.index in mobile_atoms:
			mobile_restraint.addParticle(atom.index, pos)
		else:
			nonmobile_restraint.addParticle(atom.index, pos)
			
def _build_simulation(modeller, system):
	"""Build a Simulation pinned to single-threaded CPU."""
	integrator = LangevinMiddleIntegrator(
		TEMPERATURE * unit.kelvin,
		1.0 / unit.picosecond,            # friction coefficient
		TIMESTEP * unit.picoseconds,
	)
	# Force OpenMM to single-threaded CPU to avoid clashes with multiprocessing
	# https://github.com/openmm/openmm/issues/4424
	platform = Platform.getPlatformByName('CPU')
	properties = {'Threads': '1'}
 
	simulation = Simulation(modeller.topology, system, integrator, platform, properties)
	simulation.context.setPositions(modeller.positions)
	return simulation

def _save_outputs(modeller, minimized_positions, ligand_entries, out_dir):
	"""Write minimized PDB and per-ligand SDFs."""
	pdb_modeller = Modeller(modeller.topology, minimized_positions)
	with open(out_dir / 'system_minimized.pdb', 'w') as f:
		PDBFile.writeFile(pdb_modeller.topology, pdb_modeller.positions, f)
 
	for basename, ligand_mol, atom_indices in ligand_entries:
		minimized_coords = np.array([
			minimized_positions[i].value_in_unit(unit.angstrom)
			for i in atom_indices
		]) * openff_unit.angstrom
		ligand_mol.conformers[0] = minimized_coords
		out_path = out_dir / f'{basename}_minimized.sdf'
		ligand_mol.to_file(str(out_path), file_format='SDF')

# ===================================================
# Main refinement pipeline
# ===================================================

@timeout(seconds=900)
def refine_system(input_dir):
	"""
	Structure refinement workflow:
	1. Prepare protein-only structure with PDBFixer
    2. Combine full PLI system
    3. Add additional forces
    4. Run constrained energy minimization
	"""
 
	handler = logging.FileHandler(DATA_DIR / 'logs' / f'{input_dir}.log')
	handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
	logger.addHandler(handler)
 
	if os.path.isfile(f'{DATA_DIR}/processed_systems/{input_dir}/system_minimized.pdb'):
		logger.removeHandler(handler)
		handler.close()
		return
 
	try:
		with tempfile.TemporaryDirectory() as tmp_dir:
            # ------- Step 2: Build protein-only modeller -------
			pdb_path = f'{DATA_DIR}/systems/{input_dir}/receptor.pdb'
 
			if _get_file_length(pdb_path) > 10:
				_clean_file(pdb_path)
				_rename_single_atom_residues(pdb_path)
				special_residues = _find_cofactors(pdb_path)
				modeller = _prepare_amber(tmp_dir, pdb_path, special_residues)
			else:
				modeller = Modeller(Topology(), [] * unit.nanometers)
		
			# ------- Step 3: Add ligands -------
			ligand_entries, ligand_molecules = _add_ligands_to_modeller(modeller, input_dir)
			all_atoms = list(modeller.topology.atoms())
			ligand_indices = _collect_ligand_and_cofactor_indices(modeller, ligand_entries, all_atoms)
		
			# Explicitly mark topology as non-periodic.
			modeller.topology.setPeriodicBoxVectors(None)

			system_generator = SystemGenerator(
				forcefields=FORCEFIELD_LIST,  # IMPLICIT WATER MODEL ADDED https://github.com/openmm/openmm/issues/3364
				small_molecule_forcefield='openff-2.2.0',
				molecules=ligand_molecules,
				nonperiodic_forcefield_kwargs={
					'nonbondedMethod': CutoffNonPeriodic,
					'nonbondedCutoff': NONBONDED_CUTOFF * unit.nanometer,
				}
			)

			out_dir = DATA_DIR / 'processed_systems' / input_dir
			os.makedirs(out_dir, exist_ok=True)
			with open(out_dir / 'system_protonated.pdb', 'w') as f:
				PDBFile.writeFile(modeller.topology, modeller.positions, f)

			# Remove duplicate entries: TL1 / Tl, FE2 / FE
			system_generator.forcefield = _clean_ff(system_generator.forcefield)
			system = system_generator.create_system(modeller.topology)
		
			# ------- Step 4: Identify mobile region around ligands -------
			mobile_atoms = _compute_mobile_atoms(modeller, ligand_indices, all_atoms)
		
			# ------- Step 5: Build & apply restraints -------
			# Primary: skip rebuilt atoms (they need to relax freely).
			# Fallback: rebuilt residues are already GLY, so restrain everything.
			original_path = f'{DATA_DIR}/pdb/raw/{input_dir}.pdb'
			skip_indices = _get_rebuilt_atom_indices(
				original_path, modeller.topology, modeller.positions
			)
			logger.info(f"Skipping restraints on {len(skip_indices)} rebuilt atoms")
		
			nonmobile_restraint, mobile_restraint = _build_restraints()
			_populate_restraints(
				modeller, mobile_atoms, skip_indices,
				nonmobile_restraint, mobile_restraint,
			)
			system.addForce(nonmobile_restraint)
			system.addForce(mobile_restraint)
		
			# ------- Step 6: Build simulation, save snapshot, NaN check, minimize -------
			simulation = _build_simulation(modeller, system)
			simulation.minimizeEnergy(maxIterations=MINIMIZATION_STEPS)
		
			# ------- Step 7: Save outputs -------
			state = simulation.context.getState(getEnergy=True, getPositions=True)
			_save_outputs(modeller, state.getPositions(), ligand_entries, out_dir)
 
	except Exception as e:
		print(f'{input_dir} - {e}')
		#if os.path.isdir(f'{DATA_DIR}/processed_systems/{input_dir}'):
		#	shutil.rmtree(f'{DATA_DIR}/processed_systems/{input_dir}')
		logger.exception(f"Refinement failed for {input_dir}")
 
	finally:
		logger.removeHandler(handler)
		handler.close()

# ============================================================================
# Driver
# ============================================================================
 
def safe_refine_system(input_dir):
	try:
		refine_system(input_dir)
	except TimeoutError:
		print(f'{input_dir} - timed out after 3600s, skipping')
	except Exception as e:
		print(f'{input_dir} - unhandled error: {e}')
  
if __name__ == '__main__':
	subdir_list = os.listdir(f'{DATA_DIR}/systems')
	#Parallel(n_jobs=72, verbose=10)(delayed(safe_refine_system)(input_dir) for input_dir in subdir_list)
	safe_refine_system('6aro_E')
