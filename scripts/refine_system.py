from src.config import DATA_DIR

import os
import tempfile
import numpy as np

from scipy.spatial import KDTree
from pdbfixer import PDBFixer
from openmmforcefields.generators import SystemGenerator
from openmm.app import PDBFile, Modeller, Topology, ForceField, Simulation
from openff.toolkit import Molecule
from openff.units import unit as openff_unit
from openff.nagl_models import list_available_nagl_models
from openff.nagl import GNNModel
from openmm import CustomExternalForce, LangevinMiddleIntegrator, unit, Platform
import logging
from joblib import Parallel, delayed

from concurrent.futures import TimeoutError as FuturesTimeoutError
import functools

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

STANDARD_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYM', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 
    'TRP', 'TYR', 'VAL', 'HIE', 'HIP', 'HID', 'HSD', 'HSE', 'HSP',
    'ACE', 'NME'
}

STANDARD_BASES = {'A', 'U', 'G', 'C', 'DA', 'DT', 'DG', 'DC',
	'A3', 'A5', 'U3', 'U5', 'G3', 'G5', 'C3', 'C5', 'DA3', 'DA5', 'DT3', 'DT5', 'DG3', 'DG5', 'DC3', 'DC5'
}

WATER_NAMES = {'HOH', 'WAT', 'TIP3', 'SOL', 'OPC'}
METALLOCOFACTORS = {'HEM', 'SF4', 'MGD'}
TEMPLATES_TO_REMOVE = {'AG1', 'Ce', 'Cr', 'CU1', 'EU3', 'FE2', 'TL1', 'Sm'}

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

def get_file_length(path):

	with open(path, 'r') as f:
		lines = [line.strip() for line in f]
		return len(lines)

def clean_ff(ff):
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
			if parts[0] != 'HET':
				lines_to_keep.append(line)

	with open(pdb_path, 'w') as f:
		f.write('\n'.join(lines_to_keep))

def rename_single_atom_residues(pdb_path):
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

def assign_charges_with_fallback(molecule: Molecule) -> Molecule:
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

def find_cofactors(pdb_path):
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

def add_bonds(topology, positions, resname_set):
    """Add missing bonds for residues based on interatomic distances."""
    metal_set = {'Fe', 'Mn', 'Mg', 'Ni', 'Zn', 'Cu'}

    for residue in topology.residues():
        for resname in resname_set:
            if residue.name != resname:
                continue

            atoms = list(residue.atoms())
            pos = np.array([(positions[a.index].x, positions[a.index].y, positions[a.index].z) 
                        for a in atoms])

            # Collect existing bonds to avoid duplicates
            existing_bonds = set()
            for bond in topology.bonds():
                existing_bonds.add((bond[0].index, bond[1].index))
                existing_bonds.add((bond[1].index, bond[0].index))

            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):

                    if (atoms[i].index, atoms[j].index) in existing_bonds:
                        continue

                    dist = np.linalg.norm(pos[i] - pos[j])  # already in nm from OpenMM

                    ei = atoms[i].element.symbol
                    ej = atoms[j].element.symbol

                    if ei == 'H' and ej == 'H':
                        continue

                    elif ei in metal_set and ej in metal_set:
                        continue

                    elif ei == 'H' or ej == 'H':
                        if not ei in metal_set and not ej in metal_set:
                            if dist < 0.12:
                                topology.addBond(atoms[i], atoms[j])

                    # Fe-N bonds are ~2.0 Å
                    elif ei in metal_set or ej in metal_set:
                        if dist < 0.27:  # 2.7 Å in nm
                            topology.addBond(atoms[i], atoms[j])

                    # C-C, C-N, C-O bonds are ~1.2-1.55 Å
                    elif dist < 0.18:
                        topology.addBond(atoms[i], atoms[j])

def add_seqres_with_caps(input_pdb: str, output_pdb: str):
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

def cap_dna_termini(input_pdb: str, output_pdb: str):
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

def prepare_amber(tmp_dir, pdb_path, special_residues):
	"""
	Prepare modeller and force field list for special AMBER residues
	"""

	basename = pdb_path.split('/')[-1][:-4]

	# DNA terminal renaming + 5'-phosphate stripping
	capped_path = f'{tmp_dir}/{basename}_dnacap.pdb'
	cap_dna_termini(pdb_path, capped_path)

	# Add SEQRES with caps
	seqres_path = f'{tmp_dir}/{basename}_seqres.pdb'
	add_seqres_with_caps(capped_path, seqres_path)

	Modeller.loadHydrogenDefinitions(f'{DATA_DIR}/custom_xml/protonation/special_residues_amber.xml')

	fixer = PDBFixer(seqres_path)
	fixer.findMissingResidues()
	fixer.findMissingAtoms()
	fixer.addMissingAtoms()
	fixer.addMissingHydrogens(PH)

	logging.getLogger("openff").setLevel(logging.ERROR)

	if special_residues:
		Modeller.loadHydrogenDefinitions(f'{DATA_DIR}/custom_xml/protonation/special_residues_amber.xml')
		modeller = Modeller(fixer.topology, fixer.positions)
		modeller.addHydrogens(pH=PH)
		add_bonds(modeller.topology, modeller.positions, special_residues)
	else:
		modeller = Modeller(fixer.topology, fixer.positions)

	# Remove all waters?
	#atoms_to_remove = [a for a in modeller.topology.atoms() if a.residue.name in {'HOH', 'WAT', 'TIP3', 'SOL', 'OPC', 'DOD'}]
	#if atoms_to_remove:
	#	modeller.delete(atoms_to_remove)

	return modeller

def get_rebuilt_atom_indices(original_pdb_path, topology, positions, tol_nm=0.005):
	"""
	Identify atoms belonging to residues that were rebuilt by PDBFixer.

	A residue is considered 'rebuilt' if its CA atom is not present at the same
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
				if line[12:16].strip() == 'CA':
					original_coords.append((x,y,z))

	if not original_coords:
		return set()
	
	tree = KDTree(np.asarray(original_coords))
	rebuilt_indices = set()
	for residue in topology.residues():
		# ACE/NME caps and similar have no CA — always rebuilt
		if residue.name in {'ACE', 'NME'}:
			rebuilt_indices.update(a.index for a in residue.atoms())
			continue

		# Only protein residues are candidates for CA-based matching
		if residue.name not in STANDARD_AMINO_ACIDS:
			continue

		ca = next((a for a in residue.atoms() if a.name == 'CA'), None)
		if ca is None:
			# Protein residue with no CA shouldn't happen, but be safe
			rebuilt_indices.update(a.index for a in residue.atoms())
			continue

		ca_pos = np.asarray(positions[ca.index].value_in_unit(unit.nanometer))
		dist, _ = tree.query(ca_pos, k=1)
		if dist > tol_nm:
			rebuilt_indices.update(a.index for a in residue.atoms())

	return rebuilt_indices

def _remove_rebuilt_residues(pdb_path, input_dir):

	fixer = PDBFixer(pdb_path)
	modeller = Modeller(fixer.topology, fixer.positions)

	original_path = f'{DATA_DIR}/pdb/raw/{input_dir}.pdb'
	rebuilt_indices = get_rebuilt_atom_indices(original_path, modeller.topology, modeller.positions)

	atoms_to_delete = set()

	for atom in modeller.topology.atoms():
		if atom.index in rebuilt_indices:
			atoms_to_delete.add(atom)

	modeller.delete(atoms_to_delete)

	with open(pdb_path, 'w') as f:
		PDBFile.writeFile(modeller.topology, modeller.positions, f)

@timeout(seconds=900)
def refine_system(input_dir):
	"""
	Structure refinement workflow:
	1. Protonate ligands and cofactors with dimorphite_dl
	2. Prepare protein-only structure with PDBFixer
	3. Combine full PLI system
	4. Run constrained energy minimization
	"""

	# ====================================================================
	# Step 1: Set-up
	# ====================================================================

	handler = logging.FileHandler(DATA_DIR / 'logs' / f'{input_dir}.log')
	handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
	logger.addHandler(handler)

	if os.path.isfile(f'{DATA_DIR}/processed_systems/{input_dir}/system_minimized.pdb'):
		return

	try:
		with tempfile.TemporaryDirectory() as tmp_dir:

			# ====================================================================
			# Step 2: Create protein-only structure and run PDBFixer
			# ====================================================================

			# First check for weird cofactors
			pdb_path = f'{DATA_DIR}/systems/{input_dir}/receptor.pdb'
			pdb_length = get_file_length(pdb_path)

			forcefield_list = ['amber19/protein.ff19SB.xml', 'amber19/DNA.OL21.xml', 'amber14/RNA.OL3.xml', 'amber19/opc3.xml',
				f'{DATA_DIR}/custom_xml/forcefield/HEM.xml', f'{DATA_DIR}/custom_xml/forcefield/MGD.xml', f'{DATA_DIR}/custom_xml/forcefield/SF4.xml']

			if pdb_length > 10:
				_clean_file(pdb_path)
				rename_single_atom_residues(pdb_path)  # fix single-atom LIG/UNK/UNL residues
				special_residues = find_cofactors(pdb_path)
				modeller = prepare_amber(tmp_dir, pdb_path, special_residues)
			else:
				modeller = Modeller(Topology(), [] * unit.nanometers)

			# ====================================================================
            # Step 3: Add ligands back with proper parameters
            # ====================================================================

			# Add each ligand back to the modeller
			ligand_molecules = []
			ligand_entries = []  # (basename, Molecule, list of atom indices)

			for ligand_file in sorted(os.listdir(f'{DATA_DIR}/systems/{input_dir}')):
				if ligand_file.endswith('.sdf'):
					basename = ligand_file.replace('.sdf', '')
					ligand_mol = Molecule.from_file(f'{DATA_DIR}/systems/{input_dir}/{ligand_file}', allow_undefined_stereo=True)
					ligand_mol = assign_charges_with_fallback(ligand_mol)
					ligand_molecules.append(ligand_mol)

					# Convert OpenFF positions to OpenMM unit system
					ligand_topology = ligand_mol.to_topology().to_openmm()
					ligand_positions = ligand_mol.conformers[0].to_openmm()

					# Record indices before adding (they'll start at current atom count)
					offset = modeller.topology.getNumAtoms()
					n_atoms = ligand_topology.getNumAtoms()
					modeller.add(ligand_topology, ligand_positions)

					ligand_entries.append((basename, ligand_mol, list(range(offset, offset + n_atoms))))

			# Merged set of all ligand indices (used for KDTree and restraints)
			all_atoms = list(modeller.topology.atoms())

			ligand_indices = set()
			for _, _, indices in ligand_entries:
				ligand_indices.update(i for i in indices if all_atoms[i].element.symbol != 'H')

			for residue in modeller.topology.residues():
				if residue.name in METALLOCOFACTORS:
					for atom in residue.atoms():
						if atom.element.symbol != 'H':
							ligand_indices.add(atom.index)

			system_generator = SystemGenerator(
				forcefields=forcefield_list, # IMPLICIT WATER MODEL ADDED https://github.com/openmm/openmm/issues/3364
				small_molecule_forcefield='openff-2.2.0',
				molecules=ligand_molecules,
			)

			# Remove duplicate entries: TL1 / Tl, FE2 / FE
			system_generator.forcefield = clean_ff(system_generator.forcefield)
			system = system_generator.create_system(modeller.topology)

			# ====================================================================
			# STEP 4: Identify mobile region around ligands
			# ====================================================================

			# Use KDTree for more efficient spatial lookup
			positions = modeller.positions
			all_positions_nm = np.array([positions[i].value_in_unit(unit.nanometer) for i in range(len(positions))])
			ligand_indices_list = sorted(ligand_indices)
			ligand_positions_nm = all_positions_nm[ligand_indices_list]
			ligand_tree = KDTree(ligand_positions_nm)

			distances, _ = ligand_tree.query(all_positions_nm, k=1)  # nearest ligand atom
			mobile_mask = distances <= MOBILE_RADIUS
			mobile_atoms = {i for i in np.where(mobile_mask)[0].tolist() if all_atoms[i].element.symbol != 'H'}

			# ====================================================================
            # STEP 5: Add restraints to both mobile and non-mobile atoms
            # ====================================================================

			nonmobile_restraint = CustomExternalForce("k*r^2; r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2)")
			nonmobile_restraint.addGlobalParameter('k', FIX_STRENGTH * unit.kilojoules_per_mole / unit.nanometer**2)
			nonmobile_restraint.addPerParticleParameter("x0")
			nonmobile_restraint.addPerParticleParameter("y0")
			nonmobile_restraint.addPerParticleParameter("z0")

			# Continuously differentiable energy term. Flat-bottom tethering with smoothstep function
			# Energy, force and second derivative at r=0.25 are 0.
			# Energy and force at 1.25 are 1, second derivative is 0.
			mobile_restraint = CustomExternalForce('w*('
				'step(r-u)*(1-step(r-(d+u)))*(a*(r-u)^5+b*(r-u)^4+c*(r-u)^3)' # [u, 1+u]
				'+step(r-(d+u))*d*(r-u)' # [1+u, +inf]
				'); '
				'r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2+eps)'
			)

			mobile_restraint.addGlobalParameter('w', TETHER_STRENGTH * unit.kilocalories_per_mole / unit.angstrom**2)
			mobile_restraint.addGlobalParameter('u', TETHER_FLATBOTTOM * unit.angstrom)
			mobile_restraint.addGlobalParameter('a', 3 * unit.angstrom**(-3))
			mobile_restraint.addGlobalParameter('b', -8 * unit.angstrom**(-2))
			mobile_restraint.addGlobalParameter('c', 6 * unit.angstrom**(-1))
			mobile_restraint.addGlobalParameter('d', 1.0 * unit.angstrom)
			mobile_restraint.addGlobalParameter('eps', 1e-16 * unit.nanometer**2) # Some noise needed in distance calculation, because r=0 in first minimization step blows up system
			mobile_restraint.addPerParticleParameter("x0")
			mobile_restraint.addPerParticleParameter("y0")
			mobile_restraint.addPerParticleParameter("z0")

			###
			# Read original CA coordinates. 
			# If residue's CA coordinate not in original file, it is a rebuilt residue. 
			# These get no restraints.
			###

			original_path = f'{DATA_DIR}/pdb/raw/{input_dir}.pdb'
			rebuilt_indices = get_rebuilt_atom_indices(original_path, modeller.topology, modeller.positions)
			logger.info(f"Skipping restraints on {len(rebuilt_indices)} rebuilt atoms")

			for atom in modeller.topology.atoms():
                        
				if atom.element.symbol == 'H':
					continue
                        
				if atom.index in rebuilt_indices:
					continue # rebuilt residue — let it relax freely

				pos = positions[atom.index].value_in_unit(unit.nanometers)
				if atom.index not in mobile_atoms:
					nonmobile_restraint.addParticle(atom.index, pos)
				else:
					mobile_restraint.addParticle(atom.index, pos)

			system.addForce(nonmobile_restraint)
			system.addForce(mobile_restraint)

			# ====================================================================
			# STEP 6: Run energy minimization
			# ====================================================================

			integrator = LangevinMiddleIntegrator(
				TEMPERATURE * unit.kelvin,
				1.0 / unit.picosecond, # friction coefficient
				TIMESTEP * unit.picoseconds
			)

			# Force OpenMM to use single-threaded CPU platform to prevent thread conflicts with multiprocessing https://github.com/openmm/openmm/issues/4424
			platform = Platform.getPlatformByName('CPU')
			properties = {'Threads': '1'}

			# Save original coordinates before energy minimization
			os.makedirs(DATA_DIR / 'processed_systems' / input_dir, exist_ok = True)
			pdb_path = f'{DATA_DIR}/processed_systems/{input_dir}/system_protonated.pdb'
			with open(pdb_path, 'w') as f:
				PDBFile.writeFile(modeller.topology, modeller.positions, f)

			simulation = Simulation(modeller.topology, system, integrator, platform, properties)
			simulation.context.setPositions(modeller.positions)

			# System diagnosis
			state = simulation.context.getState(getForces=True)
			forces = state.getForces(asNumpy=True).value_in_unit(
				unit.kilojoule_per_mole/unit.nanometer
			)

			fmag = np.linalg.norm(forces, axis=1)
			n_nan = np.isnan(fmag).sum()
			fmax = np.nanmax(fmag)

			if n_nan > 0 or fmax > 1e6:
				refine_system_backup(input_dir)

			else:
				simulation.minimizeEnergy(maxIterations = MINIMIZATION_STEPS)

				# ====================================================================
				# STEP 7: Save outputs
				# ====================================================================

				state = simulation.context.getState(getEnergy=True, getPositions=True)
				minimized_positions = state.getPositions()

				# Save minimized structure as PDB (protein/water only, no ligands)
				os.makedirs(DATA_DIR / 'processed_systems' / input_dir, exist_ok = True)
				pdb_modeller = Modeller(modeller.topology, minimized_positions)
				pdb_path = f'{DATA_DIR}/processed_systems/{input_dir}/system_minimized.pdb'
				with open(pdb_path, 'w') as f:
					PDBFile.writeFile(pdb_modeller.topology, pdb_modeller.positions, f)

				# Save each ligand as a separate mol2 with minimized coordinates
				for basename, ligand_mol, atom_indices in ligand_entries:
					minimized_coords = np.array([minimized_positions[i].value_in_unit(unit.angstrom) for i in atom_indices]) * openff_unit.angstrom
					ligand_mol.conformers[0] = minimized_coords

					sdf_path = f'{DATA_DIR}/processed_systems/{input_dir}/{basename}_minimized.mol2'
					ligand_mol.to_file(sdf_path, file_format='SDF')

	except Exception as e:
		print(f'{input_dir} - {e}')
		logger.exception(f"Refinement failed for {input_dir}")

	finally:
		logger.removeHandler(handler)
		handler.close()

def refine_system_backup(input_dir):
	"""
	Structure refinement workflow:
	1. Protonate ligands and cofactors with dimorphite_dl
	2. Prepare protein-only structure with PDBFixer
	3. Combine full PLI system
	4. Run constrained energy minimization
	"""

	# ====================================================================
	# Step 1: Set-up
	# ====================================================================

	if os.path.isfile(f'{DATA_DIR}/processed_systems/{input_dir}/system_minimized.pdb'):
		return

	with tempfile.TemporaryDirectory() as tmp_dir:

		# ====================================================================
		# Step 2: Create protein-only structure and run PDBFixer
		# ====================================================================

		# First check for weird cofactors
		pdb_path = f'{DATA_DIR}/systems/{input_dir}/receptor.pdb'
		pdb_length = get_file_length(pdb_path)

		forcefield_list = ['amber19/protein.ff19SB.xml', 'amber19/DNA.OL21.xml', 'amber14/RNA.OL3.xml', 'amber19/opc3.xml',
			f'{DATA_DIR}/custom_xml/forcefield/HEM.xml', f'{DATA_DIR}/custom_xml/forcefield/MGD.xml', f'{DATA_DIR}/custom_xml/forcefield/SF4.xml']

		if pdb_length > 10:
			_clean_file(pdb_path)
			rename_single_atom_residues(pdb_path)  # fix single-atom LIG/UNK/UNL residues
			_remove_rebuilt_residues(pdb_path, input_dir)
			special_residues = find_cofactors(pdb_path)
			modeller = prepare_amber(tmp_dir, pdb_path, special_residues)
		else:
			modeller = Modeller(Topology(), [] * unit.nanometers)

		# ====================================================================
        # Step 3: Add ligands back with proper parameters
        # ====================================================================

		# Add each ligand back to the modeller
		ligand_molecules = []
		ligand_entries = []  # (basename, Molecule, list of atom indices)

		for ligand_file in sorted(os.listdir(f'{DATA_DIR}/systems/{input_dir}')):
			if ligand_file.endswith('.sdf'):
				basename = ligand_file.replace('.sdf', '')
				ligand_mol = Molecule.from_file(f'{DATA_DIR}/systems/{input_dir}/{ligand_file}', allow_undefined_stereo=True)
				ligand_mol = assign_charges_with_fallback(ligand_mol)
				ligand_molecules.append(ligand_mol)

				# Convert OpenFF positions to OpenMM unit system
				ligand_topology = ligand_mol.to_topology().to_openmm()
				ligand_positions = ligand_mol.conformers[0].to_openmm()

				# Record indices before adding (they'll start at current atom count)
				offset = modeller.topology.getNumAtoms()
				n_atoms = ligand_topology.getNumAtoms()
				modeller.add(ligand_topology, ligand_positions)

				ligand_entries.append((basename, ligand_mol, list(range(offset, offset + n_atoms))))

		# Merged set of all ligand indices (used for KDTree and restraints)
		all_atoms = list(modeller.topology.atoms())

		ligand_indices = set()
		for _, _, indices in ligand_entries:
			ligand_indices.update(i for i in indices if all_atoms[i].element.symbol != 'H')

		for residue in modeller.topology.residues():
			if residue.name in METALLOCOFACTORS:
				for atom in residue.atoms():
					if atom.element.symbol != 'H':
						ligand_indices.add(atom.index)

		system_generator = SystemGenerator(
			forcefields=forcefield_list, # IMPLICIT WATER MODEL ADDED https://github.com/openmm/openmm/issues/3364
			small_molecule_forcefield='openff-2.2.0',
			molecules=ligand_molecules,
		)

		# Remove duplicate entries: TL1 / Tl, FE2 / FE
		system_generator.forcefield = clean_ff(system_generator.forcefield)
		system = system_generator.create_system(modeller.topology)

		# ====================================================================
		# STEP 4: Identify mobile region around ligands
		# ====================================================================

		# Use KDTree for more efficient spatial lookup
		positions = modeller.positions
		all_positions_nm = np.array([positions[i].value_in_unit(unit.nanometer) for i in range(len(positions))])
		ligand_indices_list = sorted(ligand_indices)
		ligand_positions_nm = all_positions_nm[ligand_indices_list]
		ligand_tree = KDTree(ligand_positions_nm)

		distances, _ = ligand_tree.query(all_positions_nm, k=1)  # nearest ligand atom
		mobile_mask = distances <= MOBILE_RADIUS
		mobile_atoms = {i for i in np.where(mobile_mask)[0].tolist() if all_atoms[i].element.symbol != 'H'}

		# ====================================================================
        # STEP 5: Add restraints to both mobile and non-mobile atoms
        # ====================================================================

		nonmobile_restraint = CustomExternalForce("k*r^2; r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2)")
		nonmobile_restraint.addGlobalParameter('k', FIX_STRENGTH * unit.kilojoules_per_mole / unit.nanometer**2)
		nonmobile_restraint.addPerParticleParameter("x0")
		nonmobile_restraint.addPerParticleParameter("y0")
		nonmobile_restraint.addPerParticleParameter("z0")

		# Continuously differentiable energy term. Flat-bottom tethering with smoothstep function
		# Energy, force and second derivative at r=0.25 are 0.
		# Energy and force at 1.25 are 1, second derivative is 0.
		mobile_restraint = CustomExternalForce('w*('
			'step(r-u)*(1-step(r-(d+u)))*(a*(r-u)^5+b*(r-u)^4+c*(r-u)^3)' # [u, 1+u]
			'+step(r-(d+u))*d*(r-u)' # [1+u, +inf]
			'); '
			'r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2+eps)'
		)

		mobile_restraint.addGlobalParameter('w', TETHER_STRENGTH * unit.kilocalories_per_mole / unit.angstrom**2)
		mobile_restraint.addGlobalParameter('u', TETHER_FLATBOTTOM * unit.angstrom)
		mobile_restraint.addGlobalParameter('a', 3 * unit.angstrom**(-3))
		mobile_restraint.addGlobalParameter('b', -8 * unit.angstrom**(-2))
		mobile_restraint.addGlobalParameter('c', 6 * unit.angstrom**(-1))
		mobile_restraint.addGlobalParameter('d', 1.0 * unit.angstrom)
		mobile_restraint.addGlobalParameter('eps', 1e-16 * unit.nanometer**2) # Some noise needed in distance calculation, because r=0 in first minimization step blows up system
		mobile_restraint.addPerParticleParameter("x0")
		mobile_restraint.addPerParticleParameter("y0")
		mobile_restraint.addPerParticleParameter("z0")	

		for atom in modeller.topology.atoms():
                        
			if atom.element.symbol == 'H':
				continue
                        
			pos = positions[atom.index].value_in_unit(unit.nanometers)
			if atom.index not in mobile_atoms:
				nonmobile_restraint.addParticle(atom.index, pos)
			else:
				mobile_restraint.addParticle(atom.index, pos)

		system.addForce(nonmobile_restraint)
		system.addForce(mobile_restraint)

		# ====================================================================
		# STEP 6: Run energy minimization
		# ====================================================================

		integrator = LangevinMiddleIntegrator(
			TEMPERATURE * unit.kelvin,
			1.0 / unit.picosecond, # friction coefficient
			TIMESTEP * unit.picoseconds
		)

		# Force OpenMM to use single-threaded CPU platform to prevent thread conflicts with multiprocessing https://github.com/openmm/openmm/issues/4424
		platform = Platform.getPlatformByName('CPU')
		properties = {'Threads': '1'}

		# Save original coordinates before energy minimization
		os.makedirs(DATA_DIR / 'processed_systems' / input_dir, exist_ok = True)
		pdb_path = f'{DATA_DIR}/processed_systems/{input_dir}/system_protonated.pdb'
		with open(pdb_path, 'w') as f:
			PDBFile.writeFile(modeller.topology, modeller.positions, f)

		simulation = Simulation(modeller.topology, system, integrator, platform, properties)
		simulation.context.setPositions(modeller.positions)

		simulation.minimizeEnergy(maxIterations = MINIMIZATION_STEPS)

		# ====================================================================
		# STEP 7: Save outputs
		# ====================================================================

		state = simulation.context.getState(getEnergy=True, getPositions=True)
		minimized_positions = state.getPositions()

		# Save minimized structure as PDB (protein/water only, no ligands)
		os.makedirs(DATA_DIR / 'processed_systems' / input_dir, exist_ok = True)
		pdb_modeller = Modeller(modeller.topology, minimized_positions)
		pdb_path = f'{DATA_DIR}/processed_systems/{input_dir}/system_minimized.pdb'
		with open(pdb_path, 'w') as f:
			PDBFile.writeFile(pdb_modeller.topology, pdb_modeller.positions, f)

		# Save each ligand as a separate mol2 with minimized coordinates
		for basename, ligand_mol, atom_indices in ligand_entries:
			minimized_coords = np.array([minimized_positions[i].value_in_unit(unit.angstrom) for i in atom_indices]) * openff_unit.angstrom
			ligand_mol.conformers[0] = minimized_coords

			sdf_path = f'{DATA_DIR}/processed_systems/{input_dir}/{basename}_minimized.mol2'
			ligand_mol.to_file(sdf_path, file_format='SDF')

def safe_refine_system(input_dir):
	try:
		refine_system(input_dir)
	except TimeoutError:
		print(f'{input_dir} - timed out after 900s, skipping')
	except Exception as e:
		# belt-and-suspenders: anything that escapes the inner try/except
		print(f'{input_dir} - unhandled error: {e}')

if __name__ == '__main__':
	subdir_list = os.listdir(f'{DATA_DIR}/systems')
	safe_refine_system('6aro_E')
	#Parallel(n_jobs = 72, verbose = 10)(delayed(safe_refine_system)(input_dir) for input_dir in subdir_list)
