"""
split_system.py
~~~~~~~~~~
Read individual SDF files from a directory, merge them, match against a
reference PDB, restore missing bonds by proximity, split into connected
components, classify each by PDB residue name, and write:
  - receptor_fixed.pdb        (PDB atoms *not* claimed by any ligand/cofactor)
  - lig_fixed_{i}.sdf         (components whose PDB residue is "LIG")
  - cof_fixed_{j}.sdf         (components with any other residue name)
  - split_ligands.sdf          (all components in a single multi-entry SDF)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import shutil
import subprocess

from src.config import DATA_DIR, SOURCE_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Common metal elements found in PDB structures
METAL_ELEMENTS: set[str] = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Th", "U",
}

METAL_ELEMENTS |= {x.upper() for x in METAL_ELEMENTS} # Add uppercase variants for better detection

METALLOCOFACTOR_LIST = ['HEM', 'SF4', 'MGD']

# Elements that indicate organic character in a metal-containing ligand
ORGANIC_ELEMENTS: set[str] = {"C", "N", "O", "S", "P"}

# Standard amino-acid 3-letter codes + common non-informative residue names
STANDARD_AA_AND_PLACEHOLDER: set[str] = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "LIG", "UNK", "UNL", "ACE", "NME"
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    """Tuneable parameters kept in one place."""

    coord_match_tol: float = 0.05       # Å – tolerance for PDB ↔ SDF matching
    proximity_bond_cutoff: float = 1.8   # Å – max distance for adding bonds
    default_bond_line: str = "  1  0  0  0  0\n"  # single-bond SDF placeholder

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Atom:
    x: float
    y: float
    z: float
    element: str
    raw_line: str  # original SDF atom line (preserves formatting)

    @property
    def coord(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z])

    @property
    def is_heavy(self) -> bool:
        return self.element != "H"

    @property
    def is_metal(self) -> bool:
        return self.element in METAL_ELEMENTS


@dataclass
class Bond:
    atom1: int  # 1-indexed (SDF convention)
    atom2: int
    rest: str   # bond-type + stereo fields from the SDF line

    def zero_indexed(self) -> tuple[int, int]:
        return self.atom1 - 1, self.atom2 - 1

    def canonical_pair(self) -> tuple[int, int]:
        """Sorted 0-indexed pair – useful as a set key."""
        a, b = self.zero_indexed()
        return (min(a, b), max(a, b))


@dataclass
class Molecule:
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)
    charges: dict[int, int] = field(default_factory=dict)  # 0-indexed atom → formal charge

    # -- querying ----------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    @property
    def heavy_atoms(self) -> list[tuple[int, Atom]]:
        """Return (index, atom) for every non-hydrogen atom."""
        return [(i, a) for i, a in enumerate(self.atoms) if a.is_heavy]

    @property
    def coord_matrix(self) -> NDArray[np.float64]:
        return np.array([[a.x, a.y, a.z] for a in self.atoms])

    def existing_bond_pairs(self) -> set[tuple[int, int]]:
        return {b.canonical_pair() for b in self.bonds}

    # -- mutating ----------------------------------------------------------

    def filter_by_coords(
        self, ref_coords: NDArray[np.float64], tol: float
    ) -> Molecule:
        """Return a new Molecule containing only atoms within *tol* of
        any point in *ref_coords*.  Each reference coordinate can only
        be claimed by a single SDF atom (closest wins) to prevent
        duplicate incorporation from overlapping input SDF files."""
        if len(ref_coords) == 0 or self.n_atoms == 0:
            return self._subset([])

        mol_coords = self.coord_matrix  # (M, 3)
        # Full pairwise distance matrix: (M_sdf, N_pdb)
        dists = np.linalg.norm(
            mol_coords[:, None, :] - ref_coords[None, :, :], axis=2
        )

        claimed_ref: set[int] = set()     # PDB indices already taken
        kept: list[int] = []

        # For each SDF atom, find its nearest PDB atom; collect all
        # candidates, then greedily assign closest-first.
        candidates: list[tuple[float, int, int]] = []  # (dist, sdf_idx, pdb_idx)
        for i in range(len(mol_coords)):
            j = int(dists[i].argmin())
            d = dists[i, j]
            if d < tol:
                candidates.append((d, i, j))

        # Sort by distance so the closest match wins each PDB atom
        candidates.sort()
        for d, sdf_idx, pdb_idx in candidates:
            if pdb_idx in claimed_ref:
                continue
            claimed_ref.add(pdb_idx)
            kept.append(sdf_idx)

        kept.sort()  # preserve original atom ordering
        return self._subset(kept)

    def add_proximity_bonds(self, cutoff: float, default_rest: str) -> int:
        """Add single bonds between unbonded heavy atoms closer than
        *cutoff* Å.  Returns the number of bonds added."""

        ALLOWED_ELEMENTS = {'C', 'N', 'O', 'S', 'P', 'B'}

        existing = self.existing_bond_pairs()
        heavy = self.heavy_atoms
        if not heavy:
            return 0

        idxs, atoms = zip(*heavy)
        elements = [a.element for a in atoms]
        coords = np.array([[a.x, a.y, a.z] for a in atoms])
        dists = np.linalg.norm(
            coords[:, None, :] - coords[None, :, :], axis=2
        )

        added = 0
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                if dists[i, j] < cutoff and elements[i] in ALLOWED_ELEMENTS and elements[j] in ALLOWED_ELEMENTS:
                    pair = (min(idxs[i], idxs[j]), max(idxs[i], idxs[j]))
                    if pair not in existing:
                        self.bonds.append(Bond(
                            atom1=pair[0] + 1,
                            atom2=pair[1] + 1,
                            rest=default_rest,
                        ))
                        existing.add(pair)
                        added += 1
        return added

    def connected_components(self) -> list[Molecule]:
        """Split into one Molecule per connected component (BFS)."""
        adj: dict[int, set[int]] = defaultdict(set)
        for bond in self.bonds:
            a, b = bond.zero_indexed()
            adj[a].add(b)
            adj[b].add(a)

        visited = [False] * self.n_atoms
        components: list[list[int]] = []

        for start in range(self.n_atoms):
            if visited[start]:
                continue
            queue = deque([start])
            visited[start] = True
            comp: list[int] = []
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            components.append(sorted(comp))

        return [self._subset(comp) for comp in components]

    # -- combining ---------------------------------------------------------

    def strip_metals(self) -> Molecule:
        """Return a new Molecule with all metal atoms (and their bonds)
        removed.  Returns self unchanged if no metals are present."""
        non_metal_indices = [i for i, a in enumerate(self.atoms) if not a.is_metal]
        if len(non_metal_indices) == self.n_atoms:
            return self  # nothing to strip
        return self._subset(non_metal_indices)

    # -- combining (continued) ---------------------------------------------

    @classmethod
    def merge(cls, molecules: list[Molecule]) -> Molecule:
        """Concatenate multiple Molecules into one, offsetting bond indices."""
        all_atoms: list[Atom] = []
        all_bonds: list[Bond] = []
        all_charges: dict[int, int] = {}
        offset = 0
        for mol in molecules:
            all_atoms.extend(mol.atoms)
            for bond in mol.bonds:
                all_bonds.append(Bond(
                    atom1=bond.atom1 + offset,
                    atom2=bond.atom2 + offset,
                    rest=bond.rest,
                ))
            for atom_idx, chg in mol.charges.items():
                all_charges[atom_idx + offset] = chg
            offset += mol.n_atoms
        return cls(atoms=all_atoms, bonds=all_bonds, charges=all_charges)

    # -- internal ----------------------------------------------------------

    def _subset(self, indices: list[int]) -> Molecule:
        """Build a new Molecule from a subset of atom indices, remapping
        bond numbering accordingly."""
        kept_set = set(indices)
        old_to_new = {old: new for new, old in enumerate(indices)}

        new_atoms = [self.atoms[i] for i in indices]
        new_bonds = [
            Bond(
                atom1=old_to_new[b.atom1 - 1] + 1,
                atom2=old_to_new[b.atom2 - 1] + 1,
                rest=b.rest,
            )
            for b in self.bonds
            if (b.atom1 - 1) in kept_set and (b.atom2 - 1) in kept_set
        ]
        new_charges = {
            old_to_new[old_idx]: chg
            for old_idx, chg in self.charges.items()
            if old_idx in kept_set
        }
        return Molecule(atoms=new_atoms, bonds=new_bonds, charges=new_charges)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

@dataclass
class PDBAtom:
    """Lightweight representation of a heavy atom from a PDB file."""
    coord: NDArray[np.float64]
    resname: str


def read_pdb_heavy_atoms(path: Path) -> list[PDBAtom]:
    """Return heavy atoms with coordinates and residue names from a PDB."""
    atoms: list[PDBAtom] = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip()
            if not element:
                element = line[12:16].strip().lstrip("0123456789")[0]
            if element == "H":
                continue
            atoms.append(PDBAtom(
                coord=np.array([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]),
                resname=line[17:20].strip(),
            ))
    return atoms


def pdb_coord_matrix(pdb_atoms: list[PDBAtom]) -> NDArray[np.float64]:
    """Stack PDBAtom coordinates into an (N, 3) array."""
    return np.array([a.coord for a in pdb_atoms])


def read_pdb_lines(path: Path) -> list[str]:
    """Return all lines of a PDB file (used for receptor filtering)."""
    with open(path) as fh:
        return fh.readlines()


def read_sdf_v2000(path: Path) -> Molecule:
    """Parse the first entry of a V2000 SDF file into a Molecule."""
    lines = path.read_text().splitlines(keepends=True)

    # Check if this is actually a proper V2000 file instead of V3000
    if 'V3000' in lines[3]:
        subprocess.run(['obabel', '-isdf', path, '-osdf', '-O', path], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        lines = path.read_text().splitlines(keepends=True)

    counts = lines[3]
    n_atoms = int(counts[0:3])
    n_bonds = int(counts[3:6])

    atoms: list[Atom] = []
    for i in range(4, 4 + n_atoms):
        ln = lines[i]
        atoms.append(Atom(
            x=float(ln[0:10]),
            y=float(ln[10:20]),
            z=float(ln[20:30]),
            element=ln[31:34].strip(),
            raw_line=ln.rstrip("\n"),
        ))

    bonds: list[Bond] = []
    for i in range(4 + n_atoms, 4 + n_atoms + n_bonds):
        ln = lines[i]
        bonds.append(Bond(
            atom1=int(ln[0:3]),
            atom2=int(ln[3:6]),
            rest=ln[6:],
        ))

    # Parse M  CHG property lines (formal charges)
    charges: dict[int, int] = {}
    for i in range(4 + n_atoms + n_bonds, len(lines)):
        ln = lines[i]
        if ln.startswith("M  END") or ln.startswith("$$$$"):
            break
        if ln.startswith("M  CHG"):
            # Format: M  CHG  n  aa1 vv1  aa2 vv2 ...
            parts = ln.split()
            n_entries = int(parts[2])
            for j in range(n_entries):
                atom_1idx = int(parts[3 + 2 * j])      # 1-indexed
                charge = int(parts[3 + 2 * j + 1])
                charges[atom_1idx - 1] = charge         # store 0-indexed

    return Molecule(atoms=atoms, bonds=bonds, charges=charges)


def read_sdf_directory(directory: Path) -> Molecule:
    """Read all .sdf files in *directory* and merge into a single Molecule.

    Files are sorted by name for reproducibility.  Each file is parsed as a
    single V2000 entry; bond indices are offset so the merged result is one
    contiguous atom/bond table.
    """
    sdf_files = sorted(directory.glob("*.sdf"))
    if not sdf_files:
        raise FileNotFoundError(f"No .sdf files found in {directory}")

    molecules: list[Molecule] = []
    for sdf_path in sdf_files:
        mol = read_sdf_v2000(sdf_path)
        log.info("  %s → %d atoms, %d bonds", sdf_path.name, mol.n_atoms, mol.n_bonds)
        molecules.append(mol)

    merged = Molecule.merge(molecules)
    log.info("Merged %d SDF files → %d atoms, %d bonds",
             len(sdf_files), merged.n_atoms, merged.n_bonds)
    return merged


def write_sdf(molecules: list[Molecule], path: Path) -> None:
    """Write one or more Molecules as separate entries in a single SDF."""
    with open(path, "w") as fh:
        for idx, mol in enumerate(molecules, start=1):
            fh.write(f"molecule_{idx}\n")
            fh.write("     split     3D\n")
            fh.write("\n")
            fh.write(
                f"{mol.n_atoms:3d}{mol.n_bonds:3d}"
                "  0  0  0  0  0  0  0  0999 V2000\n"
            )
            for atom in mol.atoms:
                fh.write(atom.raw_line + "\n")
            for bond in mol.bonds:
                fh.write(f"{bond.atom1:3d}{bond.atom2:3d}{bond.rest}")
            # Write formal charge properties (M  CHG lines, max 8 entries each)
            if mol.charges:
                charge_items = sorted(mol.charges.items())  # sorted by atom index
                for chunk_start in range(0, len(charge_items), 8):
                    chunk = charge_items[chunk_start:chunk_start + 8]
                    parts = "".join(
                        f" {atom_idx + 1:3d} {chg:3d}" for atom_idx, chg in chunk
                    )
                    fh.write(f"M  CHG{len(chunk):3d}{parts}\n")
            fh.write("M  END\n")
            fh.write("$$$$\n")
    log.info("Wrote %d entries → %s", len(molecules), path)


def _find_single_heavy_atom_residues(
    pdb_lines: list[str],
) -> set[tuple[str, str, str, str]]:
    """Identify residues that have only 1 heavy atom AND whose name is a
    standard amino acid or a placeholder (LIG / UNK / UNL).

    Returns a set of (chain, resname, resseq) tuples to exclude.
    """
    heavy_counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for line in pdb_lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip()
        if not element:
            element = line[12:16].strip().lstrip("0123456789")[0]
        if element == "H":
            continue
        resname = line[17:20].strip()
        chain = line[21:22]
        resseq = line[22:26].strip()
        heavy_counts[(chain, resname, resseq)] += 1

    to_remove: set[tuple[str, str, str]] = set()
    for key, count in heavy_counts.items():
        if count == 1 and key[1] in STANDARD_AA_AND_PLACEHOLDER:
            to_remove.add(key)

    if to_remove:
        log.info("Removing %d single-heavy-atom residue(s) from receptor PDB: %s",
                 len(to_remove),
                 [(k[1], k[2]) for k in sorted(to_remove)])
    return to_remove


def write_receptor_pdb(
    pdb_lines: list[str],
    ligand_coords: NDArray[np.float64],
    path: Path,
    tol: float = 0.05,
) -> None:
    """Write a PDB containing only lines whose coords are NOT in any ligand,
    and excluding single-heavy-atom residues with standard AA / placeholder
    names."""

    single_atom_residues = _find_single_heavy_atom_residues(pdb_lines)

    kept = 0
    prev_line = None
    atom_line = None

    with open(path, "w") as fh:
        for line in pdb_lines:
            if line.startswith(("ATOM", "HETATM")):
                coord = np.array([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])

                dists = np.linalg.norm(ligand_coords - coord, axis=1)
                if dists.size != 0 and dists.min() < tol:
                    continue  # this atom belongs to a ligand – skip

                # Skip single-heavy-atom residues with AA/placeholder names
                resname = line[17:20].strip()
                chain = line[21:22]
                resseq = line[22:26].strip()
                if (chain, resname, resseq) in single_atom_residues:
                    continue

                atom_line = line

            elif line.startswith('TER') and prev_line is not None and prev_line.startswith('TER'):
                continue

            elif line.startswith('TER') and atom_line is None:
                continue

            fh.write(line)
            prev_line = line
            kept += 1

    log.info("Wrote receptor PDB (%d lines) → %s", kept, path)

def get_component_resnames(
    mol: Molecule,
    pdb_atoms: list[PDBAtom],
    tol: float,
) -> set[str]:
    """Return the set of PDB residue names matched by this molecule."""
    pdb_coords = pdb_coord_matrix(pdb_atoms)
    resnames: set[str] = set()

    for atom in mol.atoms:
        dists = np.linalg.norm(pdb_coords - atom.coord, axis=1)
        nearest_idx = int(dists.argmin())
        if dists[nearest_idx] < tol:
            resnames.add(pdb_atoms[nearest_idx].resname)

    return resnames


def classify_molecule(
    mol: Molecule,
    pdb_atoms: list[PDBAtom],
    tol: float,
    ligand_resname: str = "LIG",
) -> str:
    """Classify a molecule as 'lig' or 'cof' based on PDB residue names.

    For each atom in *mol*, find the closest PDB atom and record its residue
    name.  If the majority of matched atoms come from residues named
    *ligand_resname*, return ``'lig'``; otherwise ``'cof'``.
    """
    pdb_coords = pdb_coord_matrix(pdb_atoms)
    lig_votes = 0
    cof_votes = 0

    for atom in mol.atoms:
        dists = np.linalg.norm(pdb_coords - atom.coord, axis=1)
        nearest_idx = int(dists.argmin())
        if dists[nearest_idx] < tol:
            if pdb_atoms[nearest_idx].resname == ligand_resname:
                lig_votes += 1
            else:
                cof_votes += 1

    label = "lig" if lig_votes >= cof_votes else "cof"
    log.info("  Classified component (%d atoms) as %s "
             "(LIG matches: %d, other: %d)",
             mol.n_atoms, label, lig_votes, cof_votes)
    return label


# ---------------------------------------------------------------------------
# Source-SDF exact-match helper
# ---------------------------------------------------------------------------

def find_matching_source_sdf(
    component: Molecule,
    sdf_dir: Path,
    tol: float,
) -> Path | None:
    """Return the path of a source SDF whose heavy-atom coordinates are an
    exact match for *component*, or ``None`` if no match is found.

    "Exact match" means:
      1. The source SDF has the same number of heavy atoms as *component*.
      2. Every atom in *component* has a counterpart in the source SDF within
         *tol* Å, and vice-versa (bijective pairing via nearest-neighbour).

    When a match is found, copying the source file is preferred over
    re-serialising through ``write_sdf`` because it preserves the original
    bond orders, stereo flags, and properties exactly.
    """
    comp_coords = component.coord_matrix  # (N, 3)
    n = len(comp_coords)
    if n == 0:
        return None

    for sdf_path in sorted(sdf_dir.glob("*.sdf")):
        try:
            src = read_sdf_v2000(sdf_path)
        except Exception:
            continue

        # Only consider heavy atoms from the source for a fair comparison
        src_coords = np.array([
            [a.x, a.y, a.z] for a in src.atoms if a.is_heavy
        ])
        if len(src_coords) != n:
            continue

        # Check every component atom has a source atom within tol (forward)
        dists_fwd = np.linalg.norm(
            src_coords[:, None, :] - comp_coords[None, :, :], axis=2
        )  # (src_n, n)
        if not np.all(dists_fwd.min(axis=0) < tol):
            continue

        # Check every source atom has a component atom within tol (backward)
        if not np.all(dists_fwd.min(axis=1) < tol):
            continue

        log.info("  Exact coord match found: %s → will copy source SDF", sdf_path.name)
        return sdf_path

    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def split_system(
    pdb_path: Path = Path("system.pdb"),
    sdf_dir: Path = Path("ligands"),
    out_dir: Path = Path("."),
    cfg: PipelineConfig = PipelineConfig(),
    ligand_resname: str = "LIG",
) -> list[Molecule]:

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load & merge ---
    log.info(f'Working on {pdb_path}')
    pdb_atoms = read_pdb_heavy_atoms(pdb_path)
    pdb_coords = pdb_coord_matrix(pdb_atoms)
    mol = read_sdf_directory(sdf_dir)
    log.info("Loaded %d PDB heavy atoms, %d merged SDF atoms, %d merged SDF bonds",
             len(pdb_atoms), mol.n_atoms, mol.n_bonds)

    # --- step 1: keep only atoms present in the PDB ---
    mol = mol.filter_by_coords(pdb_coords, tol=cfg.coord_match_tol)
    log.info("After coord filter: %d atoms, %d bonds", mol.n_atoms, mol.n_bonds)

    # --- step 2: add proximity bonds ---
    n_added = mol.add_proximity_bonds(
        cutoff=cfg.proximity_bond_cutoff,
        default_rest=cfg.default_bond_line,
    )
    log.info("Added %d proximity bonds (cutoff=%.2f Å)",
             n_added, cfg.proximity_bond_cutoff)

    # --- step 3: split by connectivity ---
    components = mol.connected_components()
    log.info("Split into %d molecules (sizes: %s)",
             len(components), [m.n_atoms for m in components])

    # --- step 4: classify & write individual SDFs ---
    lig_idx = 0
    cof_idx = 0
    non_metal_components: list[Molecule] = []
    for comp in components:
        # Strip metal atoms – they belong in the PDB, not SDFs
        comp_no_metals = comp.strip_metals()
        if comp_no_metals.n_atoms == 0:
            log.info("  Skipping component (%d atoms) – metals only", comp.n_atoms)
            continue

        resnames = get_component_resnames(
            comp_no_metals,
            pdb_atoms,
            cfg.coord_match_tol,
        )

        if any(r in METALLOCOFACTOR_LIST for r in resnames):
            continue

        # Skip organometallic ligands not in METALLOCOFACTOR_LIST:
        # if original component has metals AND at least one organic atom
        has_metal = any(a.is_metal for a in comp.atoms)
        has_organic = any(a.element in ORGANIC_ELEMENTS for a in comp.atoms)
        if has_metal and has_organic:
            log.info("  Skipping organometallic component (%d atoms, resnames=%s) "
                     "– not in METALLOCOFACTOR_LIST", comp.n_atoms, resnames)
            shutil.rmtree(out_dir)
            return

        non_metal_components.append(comp_no_metals)

        label = classify_molecule(comp_no_metals, pdb_atoms, cfg.coord_match_tol,
                                  ligand_resname=ligand_resname)

        # Prefer copying the original source SDF when coordinates match exactly,
        # so that original bond orders, stereo, and properties are preserved.
        match_src = find_matching_source_sdf(comp_no_metals, sdf_dir, cfg.coord_match_tol)

        if label == "lig":
            lig_idx += 1
            out_path = out_dir / f"lig_fixed_{lig_idx}.sdf"
        else:
            cof_idx += 1
            out_path = out_dir / f"cof_fixed_{cof_idx}.sdf"

        if match_src is not None:
            shutil.copy(match_src, out_path)
            log.info("  Copied %s -> %s (exact coord match)", match_src.name, out_path.name)
        else:
            write_sdf([comp_no_metals], out_path)

    log.info("Wrote %d ligand(s) and %d cofactor(s)", lig_idx, cof_idx)

    # Receptor PDB (system minus ligand/cofactor atoms, but keep metals)
    all_ligand_coords = np.vstack(
        [c.coord_matrix for c in non_metal_components]
    ) if non_metal_components else np.empty((0, 3))
    pdb_lines = read_pdb_lines(pdb_path)
    write_receptor_pdb(
        pdb_lines, all_ligand_coords,
        out_dir / "receptor_fixed.pdb",
        tol=cfg.coord_match_tol,
    )

    return components
