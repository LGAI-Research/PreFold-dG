# PreFold-dG

Official repository for PreFold-dG.


## Requirements

This code is tested using the following environment:

```
python==3.12
pytorch==2.6.0
numpy==2.4.2
pandas==2.2.3
scipy==1.13.1
scikit-learn==1.6.1
biopython==1.84
tqdm==4.67.1
```


## Preprocessed Data

Raw data can be preprocessed with the provided notebook `process_data.ipynb`.

We share preprocessed files for the datasets used in the paper. Due to file size, they are hosted externally:

| File | Link |
|---|---|
| `processed_data/skempi_mut.pkl` | [Download](https://drive.google.com/file/d/1nHxtEAh6ZtJJH_CUcCgJiZSSTCwLZdi9) |
| `processed_data/skempi_wt.pkl` | [Download](https://drive.google.com/file/d/18uyOFD2yrl67dg6cCkWDyMeSieClzaoW) |
| `processed_data/her2.pkl` | [Download](https://drive.google.com/file/d/1kppaABulyeT_dwF6ArfA5M5A_wG6UMuw) |


## Data Sources

Split and group files under `data/` are derived from the following sources:

| File | Source |
|---|---|
| `data/split/rdenet.csv` | [RDE-PPI](https://github.com/luost26/RDE-PPI/blob/main/data/pdbredo_splits.txt) |
| `data/split/ppiformer.csv` | [PPIRef](https://github.com/anton-bushuiev/PPIRef/blob/main/ppiref/data/splits/skempi2_iclr24_split.json) |
| `data/split/gearbind.csv` | [GearBind](https://github.com/DeepGraphLearning/GearBind/blob/main/script/process_skempi.py) |
| `data/group/prot2cplx.csv` | [PPIformer](https://github.com/anton-bushuiev/PPIformer/blob/main/notebooks/test.ipynb) |


## Usage

```bash
python main.py --config config/skempi.yaml
```

Any config option can be overridden via CLI arguments:

```bash
python main.py --config config/skempi.yaml --epochs 100
```

See [Configuration](#configuration) for the full list of options.


## Configuration

YAML config files define model architecture, training hyperparameters, and data/split settings.

| Parameter | Description |
|---|---|
| `train_data` | Training dataset name |
| `test_data` | Optional external test dataset |
| `split` | Path to fold assignment CSV (`PDB,fold`). If omitted, a random 3-fold split is generated |
| `train_folds` | List of folds to use as hold-out pool. If omitted, all folds are used |
| `test_fold` | Fixed test fold index |
| `group_map` | Optional CSV for grouping (`PDB,group`) |
| `group_min_size` | Minimum group size for group-level metrics |

See `config/` for examples: `skempi.yaml` (cross-validation), `ppiformer.yaml` (fixed test fold), `her2.yaml` (external test set).


## Citation

TBA


## License

This project is licensed under the [BSD-3-Clause-LG AI Research License](LICENSE).
