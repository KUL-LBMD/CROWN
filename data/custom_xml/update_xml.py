import parmed as pmd
from lxml import etree
from collections import OrderedDict

parm = pmd.load_file('mow.prmtop', 'mow.inpcrd')

# Map custom atom types to their element's atomic number
type_to_element = {
    'MO': 42,  # Molybdenum
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
                element_map = {'C': 6, 'N': 7, 'O': 8, 'H': 1, 'S': 16, 'F': 26, 'M': 42}
                atype.atomic_number = element_map.get(symbol, 0)

params = pmd.openmm.OpenMMParameterSet.from_structure(parm)
params.write('mgd_ff.xml')

tree = etree.parse('mgd_ff.xml')
root = tree.getroot()

# Build the Residues section from the ParmEd structure
residues_elem = etree.SubElement(root, 'Residues')

# Group atoms by residue
from collections import OrderedDict
res_dict = OrderedDict()
for atom in parm.atoms:
    res_key = (atom.residue.name, atom.residue.idx)
    if res_key not in res_dict:
        res_dict[res_key] = []
    res_dict[res_key].append(atom)

for (res_name, res_idx), atoms in res_dict.items():
    res_elem = etree.SubElement(residues_elem, 'Residue', name=res_name)
    
    # Add atoms with their types and charges
    atom_index_map = {}
    for i, atom in enumerate(atoms):
        atom_index_map[atom.idx] = i
        etree.SubElement(res_elem, 'Atom', 
                         name=atom.name, 
                         type=atom.type,
                         charge=str(atom.charge))
    
    # Add intra-residue bonds
    seen_bonds = set()
    for atom in atoms:
        for bond in atom.bonds:
            a1, a2 = bond.atom1, bond.atom2
            if a1.idx in atom_index_map and a2.idx in atom_index_map:
                bond_key = tuple(sorted((atom_index_map[a1.idx], atom_index_map[a2.idx])))
                if bond_key not in seen_bonds:
                    seen_bonds.add(bond_key)
                    etree.SubElement(res_elem, 'Bond',
                                    atomName1=a1.name,
                                    atomName2=a2.name)

# Write the updated XML
tree.write('mgd_ff.xml', pretty_print=True, xml_declaration=True, encoding='utf-8')
