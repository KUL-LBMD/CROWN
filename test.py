from openmm.app import ForceField

ff = ForceField('amber14/RNA.OL3.xml')
tpl = ff._templates['G5']

print(f"{tpl.name}: {len(tpl.atoms)} atoms")
for a in tpl.atoms:
    print(f"  {a.name:<6} type={a.type:<6} elem={a.element.symbol}")

# Internal bonds (atom-name pairs)
print("\nBonds:")
for i, j in tpl.bonds:
    print(f"  {tpl.atoms[i].name} — {tpl.atoms[j].name}")

# External bonds = atoms that bond to the *next* residue (3'-side P, etc.)
print("\nExternal bonds (atoms bonding to neighbor residues):")
for idx in tpl.externalBonds:
    print(f"  {tpl.atoms[idx].name}")
