# Decoupled GNN transfer learning

Code and data for [Achieving Robust Extrapolation in Materials Property Prediction via Decoupled Transfer Learning](https://doi.org/10.1103/pbrg-x848) (Sugiura and Mizoguchi, PRX Intelligence, 2026). Preprint: [arXiv:2602.18054](https://arxiv.org/abs/2602.18054).

This repository reproduces the formation-energy regressions in that paper. Fermi-energy prediction and the analyses in the Supplemental Material are outside this release.

The calculation keeps a graph neural network frozen as a feature extractor and fits a simple regressor to the resulting features. The networks are CGCNN, SchNet, and DimeNet++, pretrained on the Open Catalyst 2020 S2EF task. Their embeddings are concatenated and passed to ridge regression or support-vector regression. Targets and features are standardized using the non-test rows only. Each model is scored by out-of-fold predictions on the training pool and by the ensemble prediction on the test set.

Two formation-energy tables are included.

**Layered intercalation compounds (LIC).** Host–guest structures and DFT formation energies. The predicted quantity is the formation energy per atom, $E_{\mathrm{form}}$ (eV/atom). The CSV stores the formation energy per formula unit, `Ef (eV/f.u.)`. The regression target is the derived column `Ef (eV/atom)`, equal to that value divided by the number of atoms in `Host_composition` plus one. The same embeddings are evaluated under four partitions: random (`p2_rnd_rnd`), host composition (`p2_grp_grp`), the high and low tails of $E_{\mathrm{form}}$ (`p2_tar_rnd`), and those tails together with the host grouping (`p2_targrp_grp`).

**Alloys of Interest (AoI).** Alloy structures and formation energies from the Materials Project. Training uses the MP18 snapshot. The test set is the materials added in MP21, the rows with `dataset == MP21`. The target column is `formation_energy_per_atom`, reported as $E_{\mathrm{form}}$ (eV/atom).

## Requirements

Reproducing only the regression on the precomputed embeddings requires Python 3.9 or later, NumPy, pandas, scikit-learn, and matplotlib. A GPU is not required for this step. Support vector regression uses [cuML](https://docs.rapids.ai/) when that package imports successfully, and scikit-learn otherwise.

Recomputing the embeddings requires, in addition, PyTorch, PyTorch Geometric, ASE, pymatgen, and the `ocpmodels` package. This extraction code was written against the original Open Catalyst Project library. That package is no longer distributed. One project that still carries a copy of `ocpmodels`, and that still lists the pretrained checkpoints, is [mace-ocp](https://github.com/ACEsuit/mace-ocp). Other environments that provide the same library can be used in the same way.

## Installation

```bash
git clone https://github.com/nmdl-mizo/decoupled-gnn-transfer-learning.git
cd decoupled-gnn-transfer-learning
python -m pip install -e .
```

The command above installs this repository together with the packages used for the regression. Rebuilding the crystal graphs and running the pretrained networks needs further packages. Those are installed by the following command, where `[gnn]` names an optional set of dependencies declared in `pyproject.toml`:

```bash
python -m pip install -e ".[gnn]"
```

That command adds PyTorch, PyTorch Geometric, ASE, lmdb, and pymatgen. The Open Catalyst library `ocpmodels` is not among them, and neither command installs it.

## Data

The CSV files and the embeddings used for the formation-energy regressions are included.

| Path | Contents |
| --- | --- |
| `data/lic/LC_database_rev_with_cif.csv` | Layered-compound compositions, formation energies, and CIFs |
| `data/lic/embeddings/` | LIC embeddings used in the paper |
| `data/aoi/AoI_with_timestamp.csv` | Materials Project structures (`structure_as_dict`) and formation energies |
| `data/aoi/embeddings/` | AoI embeddings used in the paper |

The largest file is about 41 MB. Provenance and licenses are given in [NOTICE](NOTICE).

## Reproducing the regression task only

These commands read the embeddings shipped in `data/` and write figures and metrics under `outputs/`. The LIC default is the random partition.

```bash
python scripts/run_lic_downstream.py --output-dir outputs/lic
python scripts/run_aoi_downstream.py --output-dir outputs/aoi
```

The other three LIC partitions use the same embeddings and a different `--preset`. Give each run its own output directory.

```bash
python scripts/run_lic_downstream.py --preset p2_grp_grp --output-dir outputs/lic_grp_grp
python scripts/run_lic_downstream.py --preset p2_tar_rnd --output-dir outputs/lic_tar_rnd
python scripts/run_lic_downstream.py --preset p2_targrp_grp --output-dir outputs/lic_targrp_grp
```

| Dataset | Target | Test set |
| --- | --- | --- |
| LIC, `p2_rnd_rnd` | $E_{\mathrm{form}}$ (eV/atom) | last fold of a 5-fold split (seed 3776) |
| LIC, `p2_grp_grp` | $E_{\mathrm{form}}$ (eV/atom) | GroupKFold on `Host_composition`: each host composition lies entirely in the training–validation pool or entirely in the test set |
| LIC, `p2_tar_rnd` | $E_{\mathrm{form}}$ (eV/atom) | $E_{\mathrm{form}} > 0.1$ or $E_{\mathrm{form}} < -1.7$ eV/atom; the training–validation pool is split at random |
| LIC, `p2_targrp_grp` | $E_{\mathrm{form}}$ (eV/atom) | a host composition is placed entirely in the test set when any of its structures has $E_{\mathrm{form}} > 0.4$ or $E_{\mathrm{form}} < -2.6$ eV/atom; the training–validation pool is then split by host composition, so no host is shared with the test set |
| AoI | $E_{\mathrm{form}}$ (eV/atom) | materials added in MP21 (`dataset == MP21`); training uses MP18 |

The LIC target cuts are strict. In `p2_tar_rnd`, a structure whose $E_{\mathrm{form}}$ equals the threshold stays in the training–validation pool. In `p2_targrp_grp`, a structure at the threshold does not by itself send its host composition to the test set. If another structure of the same host composition lies beyond the threshold, the whole composition goes to the test set, including the structure at the threshold.

Hyperparameter ranges follow the paper. For ridge regression, α runs from 0 to 10 in increments of 0.05. For SVR, C takes five geometrically spaced values from 10 to 1000, and ε runs from 0 to 0.06 in increments of 0.02. The selection criterion is the out-of-fold RMSE. `--ridge-patience 5` stops the α scan after five consecutive values without an improvement; by default the full grid is searched. `--skip-svr` fits ridge regression only. `--cpu-svr` forces the scikit-learn SVR solver.

Each output directory contains `report/split_train_valid_vs_test_target_histogram.png`, a text summary of that split, parity plots, and `report/regression_metrics.csv` (MAE, RMSE, and R² for the out-of-fold and test predictions).

## Recomputing the embeddings

This step is separate from the regression on the shipped embeddings. It rebuilds the crystal graphs from the CSV, evaluates the three pretrained models, and writes new `.npy` files. The LIC structures are read from the `cif` column. The AoI structures are read from `structure_as_dict`. An LMDB file is written as an intermediate and is not required afterwards.

The checkpoints are the OC20 S2EF models trained on the All split (CGCNN, SchNet large, and DimeNet++). The script expects the file names `cgcnn_all.pt`, `schnet_all_large.pt`, and `dimenetpp_all.pt` in the directory passed to `--checkpoint-dir`. The original Open Catalyst Project page that listed those files is no longer populated. [mace-ocp](https://github.com/ACEsuit/mace-ocp) is one place that still keeps a model list for the same checkpoints.

```bash
python scripts/run_from_csv.py \
  --dataset lic \
  --checkpoint-dir checkpoints \
  --output-dir outputs/lic_from_csv

python scripts/run_from_csv.py \
  --dataset aoi \
  --checkpoint-dir checkpoints \
  --output-dir outputs/aoi_from_csv
```

`scripts/run_from_csv.py` stops if any of the three files is missing. The new embeddings are written to `work/embeddings_lic` or `work/embeddings_aoi`. The files under `data/` are left unchanged. Graphs rebuilt from a CIF or from `structure_as_dict` need not match the shipped embeddings exactly; the numbers reported in the paper were obtained from `data/*/embeddings/`.

Extraction without the subsequent regression, and construction of the train/validation/test LMDB files alone, are available as `scripts/extract_gnn_features.py` and `scripts/build_lmdb.py`. `--max-rows N` limits the LMDB to N structures and is intended only as a check that the graph conversion runs.

## Citation

```bibtex
@article{sugiura2026decoupled,
  title   = {Achieving Robust Extrapolation in Materials Property Prediction via Decoupled Transfer Learning},
  author  = {Sugiura, Tasuku and Mizoguchi, Teruyasu},
  journal = {PRX Intelligence},
  year    = {2026},
  doi     = {10.1103/pbrg-x848},
  url     = {https://doi.org/10.1103/pbrg-x848}
}
```

Please also cite the Open Catalyst Project for the pretrained models, and the data sources listed in [NOTICE](NOTICE).

## License

The code written for this paper is released under the MIT License ([LICENSE](LICENSE)). `third_party/` is derived from `ocpmodels` and remains under the MIT License in [third_party/LICENSE](third_party/LICENSE). The distributed CSV files and embeddings are CC BY 4.0. Details are in [NOTICE](NOTICE).
