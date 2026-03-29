# CROWN

Curated Repository Of Well-resolved Non-covalent interactions.

## About

CROWN is a protein–ligand interaction database containing 153,005 carefully curated structures focused on well-resolved non-covalent interactions.

The dataset was generated using a fully automated, open-source preprocessing pipeline built on PLInder, which is documented and implemented within this repository.

This repository provides:

- All scripts required to rebuild the CROWN dataset
- Metadata for all CROWN entries

The full dataset can be accessed at:
[https://crown.lbmd.be](https://crown.lbmd.be)

<img width="400" alt="image" src="https://github.com/user-attachments/assets/a6583ed2-b564-4c28-9d49-80ca6704e2e4" />

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/robin-poelmans/CROWN.git
cd CROWN
pip install -e .
```

Install the CROWN dataset:
```bash
bash download_data.sh
```

## Usage

Preprocess the CROWN training dataset:
```bash
python scripts/preprocess_crown.py
```

## License

The CROWN dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Contact

For questions or issues, contact:

Robin Poelmans
Laboratory for Biomolecular Modelling and Design, Department of Chemistry, KU Leuven
robin.poelmans@kuleuven.be
