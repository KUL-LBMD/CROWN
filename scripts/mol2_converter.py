"""Convert minimized protein-ligand complexes to MOL2 format.

For each system in DATA_DIR/processed_systems/:
  1. Separate the ligand chain from the receptor.
  2. Write receptor[_minimized].pdb and ligand[_minimized].sdf to DATA_DIR/complexes/.
  3. Convert all four files to MOL2 in DATA_DIR/mol2_files/.

Two ligand cases are handled:
  - Small-molecule ligand: chain_A_minimized.sdf exists. We locate the receptor
    chain whose nearest atom is closest to the ligand and treat it as the
    ligand chain.
  - Cofactor-style ligand (HEM, MGD, ...): chain A is the ligand by convention.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import gemmi
from rdkit import Chem
from scipy.spatial import KDTree
from joblib import Parallel, delayed

from src.config import DATA_DIR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def obabel_convert(in_fmt: str, in_path: Path, out_fmt: str, out_path: Path) -> None:
    """Run an Open Babel conversion, raising on failure."""
    result = subprocess.run(
        ['obabel', f'-i{in_fmt}', str(in_path), f'-o{out_fmt}', '-O', str(out_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Open Babel sometimes "succeeds" with a useful warning; surface it only on failure.
    if not out_path.is_file():
        raise RuntimeError(f'obabel produced no output for {in_path}: {result.stderr.strip()}')


def closest_chain_to_ligand(model: gemmi.Model, lig_tree: KDTree) -> Optional[str]:
    """Return the name of the chain whose nearest atom to the ligand is closest."""
    best_name: Optional[str] = None
    best_dist = float('inf')
    for chain in model:
        coords = [[atom.pos.x, atom.pos.y, atom.pos.z]
                  for residue in chain for atom in residue]
        if not coords:
            continue
        d = float(lig_tree.query(coords, k=1)[0].min())
        if d < best_dist:
            best_dist, best_name = d, chain.name
    return best_name


def extract_chain_as_pdb(structure: gemmi.Structure, chain_name: str, out_path: Path) -> None:
    """Write a PDB containing only `chain_name` from `structure`."""
    s = structure.clone()
    model = s[0]

    for i in range(len(model) - 1, -1, -1):
        if model[i].name != 'A':
            del model[i]

    s.write_pdb(str(out_path))

# ---------------------------------------------------------------------------
# Per-system processing
# ---------------------------------------------------------------------------

def process_system(subdir: str, tmp_dir: Path) -> None:
    data = Path(DATA_DIR)
    src = data / 'processed_systems' / subdir
    complex_dir = data / 'complexes' / subdir
    mol2_dir = data / 'mol2_files' / subdir
    complex_dir.mkdir(parents=True, exist_ok=True)
    mol2_dir.mkdir(parents=True, exist_ok=True)

    structure_min = gemmi.read_structure(str(src / 'system_minimized.pdb'))
    structure = gemmi.read_structure(str(src / 'system_protonated.pdb'))

    lig_sdf_min = src / 'chain_A_minimized.sdf'
    if lig_sdf_min.is_file():

        print('Working in ligand mode')

        # --- Case 1: SDF ligand available; identify the closest receptor chain.
        shutil.copy(data / 'systems' / subdir / 'chain_A.sdf',
                    complex_dir / 'ligand.sdf')
        shutil.copy(lig_sdf_min, complex_dir / 'ligand_minimized.sdf')

        rdmol = next(Chem.SDMolSupplier(str(lig_sdf_min), removeHs = False))
        if rdmol is None:
            raise ValueError(f'RDKit failed to parse {lig_sdf_min}')
        lig_tree = KDTree(rdmol.GetConformer().GetPositions())

        chain_name = closest_chain_to_ligand(structure_min[0], lig_tree)
        if chain_name is None:
            raise RuntimeError(f'No chain with atoms found in {subdir}')
    else:
        # --- Case 2: cofactor-style ligand (HEM, MGD, ...). Chain A is the ligand.
        chain_name = 'A'

        print('Working in cofactor mode')

        for struct, suffix, sdf_name in [
            (structure, '', 'ligand.sdf'),
            (structure_min, '_min', 'ligand_minimized.sdf'),
        ]:
            pdb_path = tmp_dir / f'{subdir}{suffix}.pdb'
            extract_chain_as_pdb(struct, chain_name, pdb_path)
            obabel_convert('pdb', pdb_path, 'sdf', complex_dir / sdf_name)

    # Receptor = full structure minus the ligand chain.
    for struct, out_name in [
        (structure_min, 'receptor_minimized.pdb'),
        (structure, 'receptor.pdb'),
    ]:
        struct[0].remove_chain(chain_name)
        struct.write_pdb(str(complex_dir / out_name))

    # MOL2 conversions.
    for sdf in ('ligand.sdf', 'ligand_minimized.sdf'):
        obabel_convert('sdf', complex_dir / sdf, 'mol2',
                       mol2_dir / sdf.replace('.sdf', '.mol2'))
    for pdb in ('receptor.pdb', 'receptor_minimized.pdb'):
        obabel_convert('pdb', complex_dir / pdb, 'mol2',
                       mol2_dir / pdb.replace('.pdb', '.mol2'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(subdir):
	with tempfile.TemporaryDirectory() as tmp:
		tmp_dir = Path(tmp)

		#if os.path.isdir(f'{DATA_DIR}/complexes/{subdir}'):
		#	file_list = os.listdir(f'{DATA_DIR}/complexes/{subdir}')
		#	if len(file_list) == 4:
		#		return

		try:
			process_system(subdir, tmp_dir)
		except Exception as e:  # noqa: BLE001 - log and continue
			shutil.rmtree(f'{DATA_DIR}/complexes/{subdir}')
			shutil.rmtree(f'{DATA_DIR}/mol2_files/{subdir}')
			print(f'{subdir} - {e}')

if __name__ == '__main__':
	systems_root = Path(DATA_DIR) / 'processed_systems'
	subdirs = sorted(p.name for p in systems_root.iterdir() if p.is_dir())
	#Parallel(n_jobs = 32, verbose = 10, backend = 'multiprocessing')(delayed(main)(subdir) for subdir in subdirs)
	main('6nlg_Q')
