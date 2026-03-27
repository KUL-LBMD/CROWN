# CROWN

Curated Repository Of Well-resolved Non-covalent interactions.

## About

CROWN is a novel protein-ligand interaction database, consisting of 153,005 curated structures.
CROWN was developed with a fully automated and open-source preprocessing pipeline built on PLInder, which is specified in this repository.

This repository contains all scripts required to re-build CROWN, as well as the metadata on all CROWN entries. The full dataset is available on:
...

<img width="837" height="659" alt="image" src="https://github.com/user-attachments/assets/a6583ed2-b564-4c28-9d49-80ca6704e2e4" />

## Installation

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
