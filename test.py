import gemmi
from collections import defaultdict


def select_conformer(structure: gemmi.Structure) -> gemmi.Structure:
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

                # Remove losing altloc atoms
                losers = {a for alt in altlocs if alt != winner for a in groups[alt]}
                residue.remove_atoms_if(lambda a: a in losers)

                # Clear altloc label on surviving atoms + shared atoms
                for atom in residue:
                    atom.altloc = '\x00'

    return structure


st = gemmi.read_structure("data/CROWN/mmCIF/raw/4oc7.cif")
st = select_conformer(st)
st.make_mmcif_document().write_file("test.cif")
