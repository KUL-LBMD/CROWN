# CROWN

**Curated Repository Of Well-resolved Non-covalent interactions**

CROWN is a protein–ligand interaction database containing 153,005 carefully curated structures focused on well-resolved non-covalent interactions. The dataset was generated using a fully automated, open-source preprocessing pipeline built on [PLInder](https://github.com/plinder-org/plinder), documented and implemented within this repository.

<img width="400" alt="CROWN overview" src="https://github.com/user-attachments/assets/a6583ed2-b564-4c28-9d49-80ca6704e2e4" />

## Resources

- **Web interface**: Browse and search the full dataset interactively at [crown.lbmd.be](https://crown.lbmd.be)
- **Bulk download**: Download the complete archive from [Zenodo](https://zenodo.org/records/19334311)
- **This repository**: All scripts required to rebuild the CROWN dataset from scratch, along with metadata for all entries

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/robin-poelmans/CROWN.git
cd CROWN
pip install -e .
```

## Usage

Preprocess the CROWN dataset:

```bash
python scripts/preprocess_crown.py
```

## Citation

If you use CROWN in your work, please cite:

CROWN: Curated Repository Of Well-resolved Noncovalent interactions
Robin Poelmans, Wout Van Eynde, Bence Bruncsics, Balint Bruncsics, Adam Arany, Yves Moreau, Arnout RD Voet
bioRxiv 2026.03.30.714168; doi: [https://doi.org/10.64898/2026.03.30.714168](https://doi.org/10.64898/2026.03.30.714168)

## License

The CROWN dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Contact

Robin Poelmans
Laboratory for Biomolecular Modelling and Design, Department of Chemistry, KU Leuven
[robin.poelmans@kuleuven.be](mailto:robin.poelmans@kuleuven.be)
