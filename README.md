# Empirical Evaluation of Temporal-Structural Leakage in GNN-Based Fraud Detection

**Gilbert Woasieworm Adjei · Fred Achi · Philip Buetey Adiboye**  
Department of Computer Science, University of Ghana  
CSCD 618 / DSCD 604 — Deep Learning, 2026

---

## Overview

This project investigates a methodological flaw in how graph neural network fraud
detection models are evaluated. Applying a random train/test split to a temporal
transaction graph allows future node embeddings to contaminate past predictions
through graph message-passing — a problem we term **temporal-structural leakage**.

We measure the exact performance drop that occurs for six model architectures when
this leakage is removed by switching to a strict chronological split with causal
edge masking enforced throughout training and evaluation. The gap between the
inflated random-split score and the honest temporal-split score is the
**Leakage Delta**, our primary result.

# Project setup
setting up virtual environment

You can run the setup.py script to set up the environment or follow the steps below to set up a virtual environment manually.
Note: if you have conda installed just use the conda environment if not follow the step below to set up virtual environment

For Mac/Linux
```bash
python3 -m venv .venv
```
Activate virtual environment
```bash
source .venv/bin/activate
```


For Windows
```bash
python3 -m venv myenv
```
Activate virtual environment
```bash
source myenv/Scripts/activate
or
#powershell
myenv\Scripts\Activate.ps1
or
#cmd
myenv\Scripts\activate.bat
```

## Requirement Installation
To install the required dependencies for this project, run the following command:

```bash
pip install -r requirements.txt
```
## Setting environment variables
To set up the environment variables for the project, do the following:
1. create a .env file in root directory of the project
2. copy the variable names in the .envexample file and paste them into the .env file
3. set the values for your kaggle api token in the variables in the .env file



---

## Dataset

**Elliptic Bitcoin Dataset** — publicly available on Kaggle.

- 203,769 transaction nodes, 234,355 directed edges
- 49 biweekly time steps
- 165 features per node (94 local transaction properties + 71 neighbourhood aggregates)
- Labels: 4,545 illicit (2.2%), 42,019 licit, 157,205 unlabelled

To add it to a Kaggle notebook: **Add data → search "elliptic data set" → select
the dataset by ellipticco**.

For local use, download via the Kaggle API:

```bash
kaggle.api.authenticate()
  kaggle.api.dataset_download_files(dataset = "ellipticco/elliptic-data-set", path=DATA_DIR)
```

---

## Project Structure

```
.
├── elliptic-fraud-leakage.ipynb   # Main experiment notebook
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│── .envexample                    # Template for your own .env (not tracked)
│── data/
│   └── results/                   # Experiment results (CSV)
│   └── figures/                   # Visualization figures (PNG)
│                                  # Raw dataset and checkpoints/ are gitignored — see below
└── src/
    ├── utils.py                   # Data loading, splits, causal_edge_index,
    │                              #   evaluation helpers, seeding
    └── models/
        ├── logistic_regression.py # Logistic Regression training
        ├── xgboosts.py            # XGBoost training
        ├── mlp.py                 # MLP (2-layer, batch norm, dropout)
        ├── gcn_graph_sage.py      # GCN and GraphSAGE (PyG-based)
        └── evolvegcn.py           # EvolveGCN — custom implementation
                                   #   (no stable PyG version available)
```

---

## Running the Experiment

Open `elliptic-fraud-leakage.ipynb` and run cells in order. The notebook is
structured as follows:

| Cells | What they do |
|---|---|
| 1–5 | Install packages, set data paths, copy src to path |
| 6–9 | Load dataset, build graph, create both splits |
| 10–17 | Train all six models under both splits, three seeds each |
| 18–22 | Aggregate results, compute Leakage Delta table |
| 23–24 | Per-seed diagnostic for unstable models (EvolveGCN, MLP) |
| 25–28 | F1 curves across all 49 time steps |
| 29–31 | XGBoost feature-importance delta (local vs neighbourhood) |
| 32–34 | GraphSAGE edge-masking ablation study |
| 35–38 | GNNExplainer subgraph visualisations |
| 39 | Save final result tables to `data/results` |
| 40–41 | Zip `checkpoints/` and `data/` into single downloadable archives |
| 42–43 | Closing interpretation notes |

---
## Key Results

| Model | F1 (random split) | F1 (temporal split) | Leakage Δ |
|---|---|---|---|
| MLP | 0.7924 ± 0.0066 | 0.2428 ± 0.2030 | 0.5496 (69.4%) |
| EvolveGCN | 0.6258 ± 0.0489 | 0.1239 ± 0.1317 | 0.5019 (80.2%) |
| Logistic Regression | 0.6034 ± 0.0168 | 0.2550 ± 0.0000 | 0.3483 (57.7%) |
| GraphSAGE | 0.7786 ± 0.0154 | 0.4691 ± 0.0184 | 0.3095 (39.7%) |
| XGBoost | 0.9590 ± 0.0026 | 0.6688 ± 0.0039 | 0.2902 (30.3%) |
| GCN | 0.6472 ± 0.0169 | 0.4302 ± 0.0231 | 0.2170 (33.5%) |

All results are averaged over three random seeds (42, 123, 456).

## Reproducing Specific Results

Cell numbers below are the exact code cell that produces each output (not the
markdown header describing it), counting from 0 as the first cell in the notebook.

**Leakage Delta table** — Cell 20 (raw) / Cell 21 (formatted, mean ± std).

**F1 per time step (Figure 1 in report)** — Cell 25 (five-model curve) and
Cell 27 (all six models, including EvolveGCN).

**Feature-importance delta (Figure 2 in report)** — Cell 29 (ungrouped) and
Cell 30 (local vs. neighbourhood grouping, the version used in the report).

**Edge-masking ablation (Figure 3 in report)** — Cell 32 (table) and
Cell 33 (the two-panel plot used in the report).

**GNNExplainer subgraphs (Figure 4 in report)** — Cell 36 (selects the
true-positive and false-negative node indices) and Cell 37 (generates and
saves both subgraph plots).

---

## Implementation Notes

**Causal edge masking:** The `causal_edge_index(data, t)` function in `src/utils.py`
filters the full edge index to retain only edges whose endpoints both appear at or
before time step `t`. This is called before every forward pass during temporal
evaluation and is the mechanism that prevents future graph topology from influencing
past predictions.

**EvolveGCN:** No stable implementation of EvolveGCN-O exists in the version of
PyTorch Geometric available on Kaggle (2.8.0.post1). We implemented the
recurrent weight-evolution layer from scratch in `src/models/evolvegcn.py`.
The GRU cell updates each row of the weight matrix conditioned on a graph-level
feature summary, then applies the evolved weight as a standard GCN convolution
with degree normalisation. Gradient-norm clipping (`max_norm=1.0`) is applied
at every training step to prevent loss divergence.

**Feature scaling:** The `StandardScaler` is fit only on the training partition
of each split, then applied to the full node feature matrix. This prevents a
second, subtler form of leakage through the scaler's own mean and variance
statistics.

**Class imbalance:** The illicit class makes up 2.2% of labelled nodes. All
models are trained with class-weighted loss. Accuracy is not reported as a
primary metric for this reason.

---

### src/
The fixed package used for all experiments in the report.
- `src/utils.py`: data loading, the two split functions (random_split, temporal_split),
  causal edge masking, seeding, and evaluation helpers.
- `src/models/logistic_regression.py`, `xgboosts.py`, `mlp.py`: the three non-graph baselines.
- `src/models/gcn_graph_sage.py`: GCN and GraphSAGE, using standard PyTorch Geometric layers.
- `src/models/evolvegcn.py`: the from-scratch EvolveGCN implementation (custom GRU-based
  recurrent weight-evolution layer), including gradient-norm clipping (max_norm=1.0) added to
  stabilise training, discussed in Section 4.5 of the report.

Not included here: the raw Elliptic Bitcoin dataset itself. It is downloaded via the Kaggle API
(`ellipticco/elliptic-data-set`) and is not redistributed in this package due to its size
(approximately 150MB) and its own Kaggle license terms.

### notebook
`elliptic-fraud-leakage.ipynb`: the full, executed notebook that produced every number and
figure in the report. Runs end to end on Kaggle with a GPU accelerator and imports directly
from `src/`. All 44 cells ran with no errors in the run this package is built from.

### data/results/
CSV exports of every quantitative result in the report.
- `final_three_seed_results.csv`: all 36 individual (model, split, seed) F1/precision/recall rows.
- `final_leakage_delta.csv` and `leakage_delta_formatted.csv`: Table 1 in the report, raw and
  formatted (mean +/- std) versions.
- `f1_by_timestep_seed42.csv`: the per-timestep F1 values behind Figure 1.
- `edge_masking_ablation.csv`: the 5-level edge-removal ablation behind Figure 3.
- `all_results.csv`, `leakage_delta.csv`: intermediate exports from earlier in the pipeline,
  kept for traceability.

### data/figures/
PNG exports of every figure in the report, at full resolution (higher than what is embedded
in the report PDF).
- `f1_all_six_models_49_timesteps.png`: Figure 1 (all six models, both splits).
- `f1_all_models_49_timesteps.png`: an earlier five-model version of the same figure, kept
  for reference.
- `feature_importance_delta_grouped.png`: Figure 2 (local vs. neighbourhood colouring).
- `feature_importance_delta.png`: the same chart before the local/neighbourhood grouping
  was added, kept for reference.
- `edge_masking_ablation_plot.png`: Figure 3.
- `gnnexplainer_TP_157325.png`, `gnnexplainer_FN_157211.png`: the two panels combined into
  Figure 4.

### checkpoints/
Not tracked in this repository (see `.gitignore`) — trained weights for all six architectures,
under both the random and temporal splits, are regenerated by running the notebook end to end.
Once generated locally, they follow this layout:

| File | Format | How to load |
|---|---|---|
| `lr_random.joblib`, `lr_temporal.joblib` | joblib | `joblib.load(path)` returns a fitted `sklearn.linear_model.LogisticRegression` |
| `xgboost_random.json`, `xgboost_temporal.json` | XGBoost native JSON | `xgb.XGBClassifier(); model.load_model(path)` |
| `mlp_random.pt`, `mlp_temporal.pt` | PyTorch state_dict | `model = MLP(165); model.load_state_dict(torch.load(path))` (class in `src/models/mlp.py`) |
| `gcn_random.pt`, `gcn_temporal.pt` | PyTorch state_dict | `model = GCN(165); model.load_state_dict(torch.load(path))` (class in `src/models/gcn_graph_sage.py`) |
| `graphsage_random.pt`, `graphsage_temporal.pt` | PyTorch state_dict | `model = GraphSAGE(165); model.load_state_dict(torch.load(path))` (same file) |
| `evolvegcn_random.pt`, `evolvegcn_temporal.pt` | PyTorch state_dict | `model = EvolveGCN(165); model.load_state_dict(torch.load(path))` (class in `src/models/evolvegcn.py`) |
