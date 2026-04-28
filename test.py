from openmm.app import ForceField

ff = ForceField('amber19/protein.ff19SB.xml',
                'amber19/DNA.OL21.xml',
                'amber19/opc3.xml')

for name in sorted(ff._templates):
    print(name)

print(f"\nTotal: {len(ff._templates)} residue templates")
