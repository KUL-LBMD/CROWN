# CROWN

**Curated Repository Of Well-resolved Non-covalent interactions**

[![Web interface](https://img.shields.io/badge/web-crown.lbmd.be-1f6feb)](https://crown.lbmd.be)
[![Data on Zenodo](https://img.shields.io/badge/data-Zenodo-1f6feb)](https://zenodo.org/records/20825315)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.03.30.714168-b31b1b)](https://doi.org/10.64898/2026.03.30.714168)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

CROWN is a protein–ligand interaction database of **178,263** carefully curated
structures focused on well-resolved non-covalent interactions. The dataset is
produced by a fully automated, open-source preprocessing pipeline, documented and
implemented in this repository.

<img width="1181" height="871" alt="CROWN overview" src="https://github.com/user-attachments/assets/beb3107f-6a2f-4488-b60d-85c82b2e9978" />

## Resources

- **Web interface** — browse and search the full dataset interactively at [crown.lbmd.be](https://crown.lbmd.be)
- **Bulk download** — the complete archive and its metadata are on [Zenodo](https://zenodo.org/records/20825315)
- **This repository** — all scripts required to rebuild CROWN from scratch

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/KUL-LBMD/CROWN.git
cd CROWN
conda env create -f environment.yml
conda activate CROWN
pip install -e .
```

## Rebuilding the dataset

To regenerate CROWN from scratch with the preprocessing pipeline:

```bash
python scripts/process_crown.py
```

## Dataset contents

The [Zenodo record](https://zenodo.org/records/20825315) contains the following files:

| File | Description |
| --- | --- |
| `crown.tar.gz` | Protein–ligand complex structures in CROWN. |
| `CROWN_metadata.parquet` | Metadata for the dataset (one row per complex). |
| `CROWN_combined_mst.parquet` | Minimum spanning trees of the CROWN cluster metrics. |
| `README.md` | Description of the structural dataset and of every metadata column. |

Unzipping the tarball with `tar -xzvf crown.tar.gz` yields the directory
`complexes/`, with one subdirectory per complex named by PDB ID and binding-chain
label. For example, `complexes/3zwe_F/` contains:

| File | Description |
| --- | --- |
| `receptor.pdb` | All protein and additive (water, ions, cofactors) chains within 4 Å of the ligand. |
| `receptor_minimized.pdb` | The same chains after energy minimization. |
| `ligand.sdf` | Protonated ligand structure. |
| `ligand_minimized.sdf` | Protonated and energy-minimized ligand structure. |

A full description of every field in `CROWN_metadata.parquet` is provided in the
`README.md` bundled with the Zenodo download.

## Clustering and minimum spanning trees

CROWN entries are clustered along four complementary similarity metrics:

| Metric | What it measures | Metadata columns | MST `metric` value |
| --- | --- | --- | --- |
| Protein sequence | Similarity of the target protein sequence(s) | `0.5/0.7/0.9 seq-sim cluster` | `seq-sim` |
| Binding pocket | Structural similarity of the binding-pocket residues | `0.5/0.7/0.9 pocketsim cluster` | `pocket-sim` |
| Ligand | Chemical similarity of the ligands | `0.5/0.7/0.9 ligsim cluster` | `lig-sim` |
| Protein–ligand interaction | Similarity of the protein–ligand interaction patterns | `0.5/0.7/0.9 pli-sim cluster` | `pli-sim` |

### Pre-computed cluster labels

For all four metrics, single-linkage cluster labels are pre-computed at similarity
cutoffs of **50%, 70%, and 90%** and stored directly in `CROWN_metadata.parquet`.
Read them off the corresponding column, e.g.:

```python
import pandas as pd

meta = pd.read_parquet("CROWN/data/metadata/CROWN_metadata.parquet")
meta["0.7 seq-sim cluster"]   # 70% protein-sequence clusters
```

### Custom thresholds via the MSTs

For thresholds other than 0.5 / 0.7 / 0.9, rebuild clusters from the minimum
spanning trees in `CROWN_combined_mst.parquet`. Each row is one MST edge:

| Column | Description |
| --- | --- |
| `metric` | Similarity metric: `seq-sim`, `pocket-sim`, `lig-sim`, or `pli-sim`. |
| `id1` | CROWN ID of the first complex. |
| `id2` | CROWN ID of the second complex. |
| `similarity` | Pairwise similarity between `id1` and `id2` for the given metric. |

To assign labels at a custom threshold, first ensure that
`CROWN_metadata.parquet` and `CROWN_combined_mst.parquet` are present in
`CROWN/data/metadata/`, then run the helper script:

```bash
python scripts/make_clusters.py label --metric seq-sim --threshold 0.4
```

You can also inspect the MST directly, for example to pull the edges of a single
metric:

```python
mst = pd.read_parquet("CROWN/data/metadata/CROWN_combined_mst.parquet")
seq_edges = mst[mst["metric"] == "seq-sim"]
```

## Citation

If you use CROWN in your work, please cite:

> CROWN: Curated Repository Of Well-resolved Non-covalent interactions.
> Robin Poelmans, Wout Van Eynde, Bence Bruncsics, Balint Bruncsics, Adam Arany,
> Yves Moreau, Arnout RD Voet. *bioRxiv* 2026.03.30.714168;
> doi: [10.64898/2026.03.30.714168](https://doi.org/10.64898/2026.03.30.714168)

## License

The CROWN dataset is licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Contact

Robin Poelmans — Laboratory for Biomolecular Modelling and Design, Department of
Chemistry, KU Leuven — [robin.poelmans@kuleuven.be](mailto:robin.poelmans@kuleuven.be)
