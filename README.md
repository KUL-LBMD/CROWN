# CROWN

**Curated Repository Of Well-resolved Non-covalent interactions**

CROWN is a protein–ligand interaction database containing 141,261 carefully curated structures focused on well-resolved non-covalent interactions. 
The dataset was generated using a fully automated, open-source preprocessing pipeline, documented and implemented within this repository.

<img width="638" height="502" alt="image" src="https://github.com/user-attachments/assets/66afa888-fc0d-411d-bb1f-22361a3abd6c" />

## Resources

- **Web interface**: Browse and search the full dataset interactively at [crown.lbmd.be](https://crown.lbmd.be)
- **Bulk download**: Download the complete archive and its associated metadata from [Zenodo](https://zenodo.org/records/19334311)
- **This repository**: All scripts required to rebuild the CROWN dataset from scratch.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/KUL-LBMD/CROWN.git
cd CROWN
conda env create -f environment.yml
conda activate CROWN
pip install -e .
```

## Usage

Preprocess the CROWN dataset:

```bash
python scripts/process_crown.py
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
