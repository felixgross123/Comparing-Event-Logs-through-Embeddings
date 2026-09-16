# Comparison of Event Logs through Embeddings

Implementation and evaluation code for the paper *Comparison of Event Logs through Embeddings* —
Felix Carl Gross, Sander J. J. Leemans, Jorge Munoz-Gama, Luciano Hidalgo, José Luis Jara.

## The technique

Two event logs are compared through representations learned from the logs themselves, rather than
through a predefined abstraction or their empirical trace distributions. One embedding model is
trained per log, and both are tied to a shared reference — the **compass**, an output matrix fitted
on the joint log and then frozen — so that the two embedding spaces are comparable.

The aligned models are read in two steps:

- **Where the logs differ.** The cross-log cosine similarity between a shared activity's two
  embedding vectors ranks it from *stable* (its contexts remain similar) to *shifted* (its contexts
  differ). An activity exclusive to one log is matched to the activity occurring in the most
  similar contexts in the other.
- **How they differ.** For one and the same context, the two models' predicted target
  distributions show which activities each log expects to occur next.

Activities sharing a timestamp are summed within an activity block instead of being ordered, so
weakly ordered logs need no imposed order. The same model covers resources, time windows and
attribute values.

## Layout

| Path | |
|---|---|
| `01. experiment/log_comparison_cwindow.py` | the technique for totally ordered logs |
| `01. experiment/00_preprocessing.ipynb` | builds the log pairs from the raw BPIC exports |
| `01. experiment/01_validity.ipynb` | validity: shifted vs. stable activities against EMSC |
| `01. experiment/01_stochasticConformance.ipynb` | per-activity shift against stochastic conformance |
| `01. experiment/02_reliability.ipynb` | reliability: ten seeds per log pair |
| `01. experiment/03_efficiency/` | feasibility: runtime benchmark, submitted as a SLURM job |
| `02. case-study/log_comparison_block.py` | the technique for weakly ordered logs, plus time windows and case attributes |
| `02. case-study/00_preprocess.ipynb` | builds the on-time and late student cohorts |
| `02. case-study/01_control-flow.ipynb`, `02_time.ipynb`, `03_data.ipynb` | the three perspectives of the case study |
| `*/XX_dimension*.ipynb` | post-study experiments w.r.t. the embedding dimension |
| `*/results/` | one result row per log pair, plus the summary CSVs |

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.14 with PyTorch and pm4py; EMSC runs through the Ebi framework (`ebi-pm`).
`01. experiment/03_efficiency/` carries its own `requirements.txt` for the cluster.

## Data

- **BPI Challenge 2011, 2015 and 2018** are not in the repository (2.8 GB). Download the XES files
  into `01. experiment/logs/raw/` and run `00_preprocessing.ipynb`, which writes the sublogs the
  experiments read.
- The **student journey** log of the case study is included, under `02. case-study/data/`.

## Running

The experiment notebooks have a `RECOMPUTE` flag and the case-study notebooks a
`RECOMPUTE_EMBEDDINGS` flag. Left at `False`, they read the results already on disk; set to `True`,
they retrain. Trained models are cached under `results/artefacts/`.

The runtime benchmark is run separately:

```sh
cd "01. experiment/03_efficiency" && sbatch run_timing.sh
```

It times the embedding comparison for all 17 log pairs; `RUN_EMSC` in `timing_benchmark.py` adds
the EMSC timings.

## Settings

Embedding dimension `d = 32`; window size `c = 4` for the experiments and `c = 2` for the case
study; Adam at a learning rate of `1e-2`, converging within 500–1500 epochs depending on log size.

## Results reported in the paper

Over 17 log pairs from the three real-life logs:

- **Valid** — a stable activity's local behaviour conforms more closely across the logs than a
  shifted one's in 7839 of 8249 comparisons (95.0 %), with no log pair below 75 %.
- **Reliable** — across ten random initialisations the mean pairwise deviation of the similarities
  is at most 0.052 and Spearman's rank correlation at least 0.901.
- **Feasible** — 81.3 ± 48.8 s per pair on a 24-core Intel Xeon 8468, 3.0 ± 0.9 s on an NVIDIA H100.

The case study compares on-time and late graduates of a computer science programme at a Latin
American university over the control-flow, time and data perspectives.
