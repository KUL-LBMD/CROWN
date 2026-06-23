import parmed as pmd

parm = pmd.load_file('heme.prmtop', 'heme.inpcrd')

# Map custom atom types to their element's atomic number
type_to_element = {
    'FE': 26,  # Iron
    'NP': 7,   # Nitrogen
    'NO': 7,   # Nitrogen
    'CY': 6,   # Carbon
    'CX': 6,   # Carbon
    'CD': 6,   # Carbon
    'LC': 6,   # Carbon (CO ligand)
    'LO': 8,   # Oxygen (CO/O2 ligand)
}

seen = set()
for atom in parm.atoms:
    atype = atom.atom_type
    if id(atype) not in seen:
        seen.add(id(atype))
        if atype.atomic_number <= 0:
            if atype.name in type_to_element:
                atype.atomic_number = type_to_element[atype.name]
            else:
                symbol = atype.name[0].upper()
                element_map = {'C': 6, 'N': 7, 'O': 8, 'H': 1, 'S': 16, 'F': 26}
                atype.atomic_number = element_map.get(symbol, 0)

params = pmd.openmm.OpenMMParameterSet.from_structure(parm)
params.write('heme_ff.xml')
