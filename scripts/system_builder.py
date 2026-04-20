"""Extract small chains from PDB files as SDF ligands.
 
For each chain in the input PDB with
 
    * strictly fewer than 100 heavy atoms, and
    * elements drawn only from {C, N, O, S, P, F, Cl, Br, I, B},
 
an SDF file is produced. Intra-residue bonds are taken from the wwPDB
Chemical Component Dictionary (``components.cif``); consecutive residues
within a chain are joined by a single bond between their two closest
heavy atoms. The remaining chains are written to a receptor PDB.
 
Output layout:
    output_dir/<pdb_stem>/
        chain_<id>.sdf
        ...
        receptor.pdb
 
Usage:
    python extract_chain_ligands.py <file.pdb> [<file2.pdb> ...] [-o out/]
"""

from src.config import DATA_DIR
from src.CROWN.ccd_cache import CCD_CACHE_PATH, normalize
from src.dimorphite_dl import dimorphite_dl as dl

import numpy as np
import pandas as pd
import gemmi
import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from typing import Iterable
import shutil
from joblib import Parallel, delayed
import functools


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
ALLOWED_ELEMENTS: set[str] = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B"}
MAX_HEAVY_ATOMS: int = 100  # chain must have strictly fewer than this
INTER_RESIDUE_CUTOFF: float = 2.2  # Å; no bond added beyond this distance
SKIP_RESIDUES = {'HEM', 'MGD', 'SF4'}
 
# Cache keyed by residue name -> {"atoms": {name: element}, "bonds": [(a1, a2, order)]}
CCD_ATOMS_BONDS_CACHE_PATH: str = os.path.join(
    os.path.dirname(CCD_CACHE_PATH), "components_atoms_bonds.pkl"
)
 
BOND_ORDER_MAP: dict[str, Chem.BondType] = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
    "AROM": Chem.BondType.AROMATIC,
    "QUAD": Chem.BondType.QUADRUPLE,
}

PH = 7.4

# Module-level global — no decorators, no pickle issues
_CCD_CACHE = None

def _get_ccd_cache():
    global _CCD_CACHE
    if _CCD_CACHE is None:
        _CCD_CACHE = build_ccd_atoms_bonds_cache()
    return _CCD_CACHE

# ---------------------------------------------------------------------------
# CCD cache with bonds
# ---------------------------------------------------------------------------
 
def build_ccd_atoms_bonds_cache(force: bool = False) -> dict:
    """Build {res_name: {'atoms': {atom_name: element}, 'bonds': [(a1, a2, order)]}}.
 
    Only heavy atoms are kept; bonds that reference excluded atoms are dropped.
    """
    if not force and os.path.exists(CCD_ATOMS_BONDS_CACHE_PATH):
        with open(CCD_ATOMS_BONDS_CACHE_PATH, "rb") as f:
            return pickle.load(f)
 
    components_cif = os.path.join(os.path.dirname(CCD_CACHE_PATH), "components.cif")
    if not os.path.exists(components_cif):
        raise FileNotFoundError(
            f"components.cif not found at {components_cif}. "
            "Download it from "
            "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
        )
 
    doc = gemmi.cif.read(components_cif)
 
    cache: dict = {}
    for block in doc:
        atom_table = block.find("_chem_comp_atom.", ["atom_id", "type_symbol"])
        if not atom_table:
            continue
 
        atoms: dict[str, str] = {}
        for row in atom_table:
            elem = row[1]
            if elem in ("H", "D"):
                continue
            name = normalize(row[0])
            # Normalise element casing: "CL" -> "Cl", "BR" -> "Br"
            atoms[name] = elem[0].upper() + elem[1:].lower() if len(elem) > 1 else elem
 
        bonds: list[tuple[str, str, str]] = []
        bond_table = block.find(
            "_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order"]
        )
        if bond_table:
            for row in bond_table:
                a1 = normalize(row[0])
                a2 = normalize(row[1])
                if a1 in atoms and a2 in atoms:
                    bonds.append((a1, a2, row[2]))
 
        cache[block.name] = {"atoms": atoms, "bonds": bonds}
 
    os.makedirs(os.path.dirname(CCD_ATOMS_BONDS_CACHE_PATH), exist_ok=True)
    with open(CCD_ATOMS_BONDS_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache

# ---------------------------------------------------------------------------
# Chain filtering
# ---------------------------------------------------------------------------
 
def chain_passes_filter(chain: gemmi.Chain) -> tuple[bool, int, set[str]]:
    """Return (passes, heavy_atom_count, elements_seen)."""
    count = 0
    elements: set[str] = set()
    for residue in chain:
        if residue.name in SKIP_RESIDUES:
            return False, count, elements
        
        for atom in residue:
            elem = atom.element.name
            if elem in ("H", "D"):
                continue
            count += 1
            elements.add(elem)

    passes = (
        0 < count < MAX_HEAVY_ATOMS
        and elements.issubset(ALLOWED_ELEMENTS)
    )
    return passes, count, elements

# ---------------------------------------------------------------------------
# RDKit molecule construction
# ---------------------------------------------------------------------------
 
def _sanitize_with_fallback(mol: Chem.Mol, label: str) -> Chem.Mol:
    """Try strict sanitization; fall back to relaxed; on failure, leave raw."""
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e1:
        try:
            flags = (
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            )
            Chem.SanitizeMol(mol, sanitizeOps=flags)

        except Exception as e2:
            return None
    return mol

def build_rdkit_mols_from_chain(
    chain, ccd_cache: dict, chain_id: str
) -> list[Chem.Mol]:
    """Build one RDKit ``Mol`` per connected fragment of a gemmi chain.
 
    * Intra-residue bonds come from the CCD's ``_chem_comp_bond`` table.
    * Consecutive residues are joined by a single bond between their two
      closest heavy atoms, but only if that minimum distance is below
      ``INTER_RESIDUE_CUTOFF``. A chain that remains in multiple connected
      components is returned as multiple ``Mol`` objects.
    """
    rwmol = Chem.RWMol()
    positions: list[tuple[float, float, float]] = []
    # Per-residue: list of (atom_name, rdkit_idx, np.array(pos)).
    residues_data: list[list[tuple[str, int, np.ndarray]]] = []
 
    residues = list(chain)
 
    # First pass: initialize atoms as point cloud
    for res_i, residue in enumerate(residues):
        per_res: list[tuple[str, int, np.ndarray]] = []
        for atom in residue:
            elem = atom.element.name
            if elem in ("H", "D"):
                continue
 
            rdatom = Chem.Atom(elem)
            idx = rwmol.AddAtom(rdatom)
            positions.append((atom.pos.x, atom.pos.y, atom.pos.z))
            per_res.append(
                (normalize(atom.name), idx, np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
            )
        residues_data.append(per_res)
 
    # -- Intra-residue bonds from CCD --
    for res_i, residue in enumerate(residues):
        template = ccd_cache.get(residue.name)
        if template is None:
            return None
        
        name_to_idx = {name: idx for name, idx, _ in residues_data[res_i]}
        for a1, a2, order in template["bonds"]:
            i1 = name_to_idx.get(a1)
            i2 = name_to_idx.get(a2)
            if i1 is None or i2 is None:
                continue
            if rwmol.GetBondBetweenAtoms(i1, i2) is not None:
                continue
            rwmol.AddBond(i1, i2, BOND_ORDER_MAP.get(order, Chem.BondType.SINGLE))
 
    # -- Inter-residue bonds: single bond between the closest atom pair,
    #    but only if that minimum distance is below the cutoff. --
    for res_i in range(len(residues_data) - 1):
        atoms_i = residues_data[res_i]
        atoms_j = residues_data[res_i + 1]
        if not atoms_i or not atoms_j:
            continue
        coords_i = np.stack([p for _, _, p in atoms_i])
        coords_j = np.stack([p for _, _, p in atoms_j])
        d = np.linalg.norm(coords_i[:, None, :] - coords_j[None, :, :], axis=-1)
        flat = int(np.argmin(d))
        min_dist = float(d.flat[flat])
        if min_dist > INTER_RESIDUE_CUTOFF:
            continue
        ii, jj = np.unravel_index(flat, d.shape) # Select i,j pair with closest distance
        idx_i = atoms_i[ii][1] # Map back to RDKit indices
        idx_j = atoms_j[jj][1]
        if rwmol.GetBondBetweenAtoms(idx_i, idx_j) is None:
            rwmol.AddBond(idx_i, idx_j, Chem.BondType.SINGLE)
 
    # -- Attach 3D conformer --
    conf = Chem.Conformer(rwmol.GetNumAtoms())
    for i, (x, y, z) in enumerate(positions):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    rwmol.AddConformer(conf, assignId=True)
 
    mol = rwmol.GetMol()
    # Split into connected components. sanitizeFrags=False so we can apply
    # our own fallback logic per fragment.
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    for i, frag in enumerate(frags):
        _sanitize_with_fallback(frag, label=f"chain {chain_id} frag {i}")
    return frags

# ---------------------------------------------------------------------------
# Receptor construction
# ---------------------------------------------------------------------------
 
def _remove_chains_by_name(model: gemmi.Model, chain_names: Iterable[str]) -> None:
    """Remove every chain in ``model`` whose name is in ``chain_names``.
 
    Uses ``remove_chain`` in a loop because gemmi can hold multiple chain
    fragments with the same name (e.g. when an ATOM/HETATM run is split
    by a TER record).
    """
    targets = set(chain_names)
    while True:
        victim = next((ch.name for ch in model if ch.name in targets), None)
        if victim is None:
            return
        model.remove_chain(victim)

#----------------------------------------------------------------------------
# Protonate ligands
#----------------------------------------------------------------------------

def protonate_ligand(mol: Chem.Mol) -> Chem.Mol:
    """
    Protonate RDMol at given pH using dimorphite
    """

    try:

        # Strip existing Hs, protonate at target pH
        mol_noh = Chem.RemoveAllHs(mol)
        protonated = dl.run_with_mol_list([mol_noh], min_ph = PH, max_ph = PH,
		pka_precision=0.0, silent=True
	    )[0]

        # Transfer 3D coordinates from original mol via substructure match
        protonated = AllChem.AssignBondOrdersFromTemplate(protonated, mol_noh)

        # Add explicit Hs with 3D coords
        protonated_h = Chem.AddHs(protonated, addCoords=True)

        return protonated_h

    except Exception as e:
        print(e)
        return None

# ---------------------------------------------------------------------------
# Per-PDB driver
# ---------------------------------------------------------------------------
 
def process_pdb(basename: str) -> None:

    ccd_cache = _get_ccd_cache()          # <-- load inside the worker

    structure = gemmi.read_structure(f'{DATA_DIR}/pdb/fixed/{basename}.pdb')
    model = structure[0]
 
    out_dir = f'{DATA_DIR}/systems/{basename}'
    os.makedirs(out_dir, exist_ok = True)
 
    extracted_chain_names: list[str] = []
    for chain in list(model):  # materialise: we may delete from model later
        passes, n_heavy, elements = chain_passes_filter(chain)
        if passes:

            if not elements.issubset(ALLOWED_ELEMENTS):

                print(f'{basename} - {chain.name} - Bad elements')

                shutil.rmtree(out_dir)
                return
            
            mols = build_rdkit_mols_from_chain(chain, ccd_cache, chain.name)
            if mols is None:

                print(f'{basename} - {chain.name} - Molecule building failed')

                shutil.rmtree(out_dir)
                return
 
            for frag_i, mol in enumerate(mols):
                if len(mols) == 1:
                    stem = f"chain_{chain.name}"
                else:
                    stem = f"chain_{chain.name}_frag{frag_i}"

                mol_h = protonate_ligand(mol)

                if mol_h is None:
                    print(f'{basename} - {chain.name} - Protonation failed')
                    shutil.rmtree(out_dir)
                    return

                mol_h.SetProp("_Name", f"{basename}_{stem}")
                sdf_path = f"{out_dir}/{stem}.sdf"
                with Chem.SDWriter(str(sdf_path)) as writer:
                    writer.write(mol_h)
                    
            extracted_chain_names.append(chain.name)
 
    # Write receptor: original structure minus the extracted chains.
    receptor = structure.clone()
    _remove_chains_by_name(receptor[0], extracted_chain_names)
    receptor_path = f'{out_dir}/receptor.pdb'
    receptor.write_pdb(str(receptor_path))

def main():
    df = pd.read_csv(f'{DATA_DIR}/metadata/pli_filter_pass.csv')
    basenames = df['filename'].tolist()

    # ensure the cache file exists BEFORE spawning workers
    build_ccd_atoms_bonds_cache()

    Parallel(n_jobs = 32, verbose = 10)(delayed(process_pdb)(basename) for basename in basenames)

if __name__ == '__main__':
    main()
