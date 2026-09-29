# Community detection: Louvain, Leiden, Infomap, Bayesian SBM, Spectral

**Our question:** which method finds the true communities in a network, and when does each one stop working?

We test 5 algorithms from 4 different paradigms on synthetic LFR graphs, where we know the answer and control the difficulty. Then we check whether the lessons hold on a real network (email-Eu-core).

| Part | Algorithm(s) | Paradigm | File |
|---|---|---|---|
| 1 | Louvain + Leiden (one person) | Modularity optimisation (greedy) | `louvain.py`, `leiden.py` |
| 2 | Infomap | Information flow (random walks) | `infomap.py` |
| 3 | Bayesian SBM | Statistical inference (generative model) | `sbm.py` |
| 4 | Spectral clustering | Linear algebra + K-means | `spectral.py` |

> **Rule:** everyone uses the shared harness. Nobody writes their own loading, scoring or plotting code. That's what keeps our results comparable. The report structure is in `REPORT_GUIDE.txt`.

---

## Quick start (Colab)

```python
!git clone <OUR_REPO_URL>
%cd community-detection
!pip install -q -r requirements.txt

!python run.py louvain smoke     # 10 seconds, must say PASSED
!python run.py louvain all       # every experiment for your algorithm
```

Then commit the new CSVs in `results/raw/` (or download them and upload to GitHub).
Algorithm names: `louvain`, `leiden`, `infomap`, `sbm`, `spectral`.

**How many runs?** There are 2 *graph* seeds (`s1`, `s2` = two random versions of each LFR graph, already built) and 5 *algorithm* seeds (each algorithm runs 5 times per graph, automatically). That's 40 LFR runs + 5 email runs per algorithm. `make_figures.py` does all the averaging. See the box at the top of `REPORT_GUIDE.txt`.

**After your runs:** follow `EVALUATION_PROMPT.txt` to analyse your results in the shared format.

The first run automatically downloads and cleans email-Eu-core from SNAP, which needs internet (Colab has it).

### Installing graph-tool (Bayesian SBM only)

graph-tool isn't on pip, so only the SBM owner needs this. The easiest route in Colab is conda:

```python
!pip install -q condacolab
import condacolab; condacolab.install()          # the runtime restarts, which is expected
# then, in a NEW cell after the restart:
!mamba install -q -y -c conda-forge graph-tool
```

On your own laptop, use `conda install -c conda-forge graph-tool` (Mac/Linux; on Windows use WSL). On Ubuntu 24.04 you can use `sudo apt install python3-graph-tool` with the system Python 3.12. Check it worked with `python -c "import graph_tool.all"`.

---

## What's in the repo

```
community-detection/
├── README.md              ← you are here
├── REPORT_GUIDE.txt        ← report structure, section templates, who writes what
├── EVALUATION_PROMPT.txt   ← prompt for analysing your results after your runs
├── requirements.txt
├── run.py                 ← run experiments:  python run.py <algorithm> <experiment>
├── make_figures.py        ← builds every figure + summary table from results/raw/
├── harness/core.py        ← shared loading, timing, scoring, seeds. DON'T EDIT
├── algorithms/
│   ├── _template.py       ← the 5 things every algorithm file must have
│   ├── louvain.py
│   ├── leiden.py
│   ├── infomap.py
│   ├── sbm.py
│   └── spectral.py
├── data/
│   ├── lfr/               ← 8 synthetic graphs + summary.csv (already included)
│   └── email/             ← created automatically on first run
├── scripts/
│   ├── generate_lfr.py    ← rebuilds data/lfr exactly (fixed seeds)
│   └── prepare_email.py   ← downloads + cleans email-Eu-core
├── results/
│   ├── raw/               ← one CSV per algorithm per experiment (commit these!)
│   └── partitions/        ← each run's found communities (for failure analysis)
└── figures/               ← output of make_figures.py
```

---

## The datasets

Every graph folder contains the same two files:

- `edges.txt` has one link per line, written as `u v`. This is the only thing the algorithm sees.
- `labels.txt` has one line per node, written as `node true_community`. It's the answer key and is only used for scoring.

### LFR synthetic graphs (`data/lfr/`)

μ is the share of each node's links that go to *other* communities, so a higher μ means harder to find.

| Graph | n | μ | Purpose |
|---|---|---|---|
| `mix_mu0.2_s1`, `_s2` | 1,000 | 0.2 | **Easy.** Every method should score close to 1. If yours doesn't, suspect a bug. |
| `mix_mu0.4_s1`, `_s2` | 1,000 | 0.4 | **Medium.** Differences should start to show. |
| `mix_mu0.6_s1`, `_s2` | 1,000 | 0.6 | **Hard.** Communities are barely there. Find out who still finds them. |
| `scale_n10000` | 10,000 | 0.4 | **Speed test, medium.** |
| `scale_n50000` | 50,000 | 0.4 | **Speed test, large.** A crash or out-of-memory counts as a result, so record it. |

`s1` and `s2` are two random versions of the same setting, and we average them. Graphs with the same seed have identical degrees and community sizes and differ **only in μ**, so any change in score is caused by mixing alone.

Settings: τ1 = 3, τ2 = 1.5, average degree 20, max degree 50, community sizes 20 to 100 (Lancichinetti et al., 2008). They were generated with NetworKit, because networkx's generator overshoots μ by 0.1 to 0.2. The actual μ is within 0.02 of each target (see `summary.csv`).

### email-Eu-core (`data/email/`)

These are emails between members of a European research institution, and the ground truth is each person's department (42 departments). `prepare_email.py` makes the graph undirected, removes self-loops and duplicate edges, and keeps the largest connected component. Its `summary.csv` also records the network's **own μ**, which lets us place it on the LFR difficulty curve (Figure 6).

---

## How the harness works

Each algorithm file provides **one function** plus 4 settings:

```python
NAME = "Louvain"
PARAMS = {"resolution": 1.0}                       # defaults
SWEEP = {"resolution": [0.25, 0.5, 1, 2, 3]}       # values for the parameter experiment
VARIANTS = {"default": {}, "first_level_only": {...}}  # ablation switches

def detect(g, seed, **params):
    return membership   # membership[i] = community of node i
```

The harness does everything else **identically for all 5 algorithms**:

1. loads the graph as an igraph `Graph`, hiding the labels
2. calls `detect()` in a separate process and times **only** that call
3. stops any run that takes longer than **15 minutes** (`--timeout=SECONDS` to change) and records it as `timeout`
4. scores the result (see the metrics below)
5. repeats with 5 algorithm seeds (0 to 4). If a graph times out or runs out of memory, the remaining seeds for that graph are skipped
6. saves one row per run to `results/raw/`
7. records crashes, timeouts and out-of-memory errors as rows instead of stopping, because they're results too

All implementations are compiled or vectorised (igraph, leidenalg, infomap, graph-tool, scipy/scikit-learn), so runtime comparisons are fair.

**Spectral clustering needs to be told K.** It sets `NEEDS_K = True`, so the harness passes it the true number of communities. That's an advantage the other methods don't get, and the report must say so. The `eigengap_k` variant estimates K without the answer key, which shows what happens in real use.

### Experiments

| Command | What it runs | Output | Feeds |
|---|---|---|---|
| `smoke` | Easy graph, 1 seed, checks NMI > 0.9 | printed PASS/FAIL | sanity check |
| `main` | All 9 graphs × 5 seeds, default settings | `main_<algo>.csv`, `stability_<algo>.csv` | Figs 1, 2, 3, 4, 6, 8, 9, 10 |
| `sweep` | Each `SWEEP` value on `mix_mu0.4_s1` + email × 5 seeds | `sweep_<algo>.csv` | Fig 5 |
| `ablation` | Each `VARIANTS` entry on μ = 0.2 / 0.4 / 0.6 + email × 5 seeds | `ablation_<algo>.csv` | Fig 7 |
| `all` | All of the above | | |

Add `--skip-large` to leave out the 50k graph while testing.

**Rough run times for `all`:** Louvain, Leiden and Infomap take about 1 to 2 minutes each. Bayesian SBM and Spectral take longer (up to about an hour), mostly because of the 50k graph. Start those early and keep the Colab tab open.

### What each algorithm tunes and ablates

| Algorithm | `SWEEP` (parameter experiment) | `VARIANTS` (ablation) |
|---|---|---|
| Louvain | resolution γ | full vs first level only (no aggregation) |
| Leiden | resolution γ | 2 passes vs 1 pass vs until converged. Louvain vs Leiden as a whole is the refinement ablation |
| Infomap | Markov time | two-level vs hierarchical; 1 vs 10 restarts |
| Bayesian SBM | none (K is inferred automatically) | degree-corrected vs plain; flat vs nested |
| Spectral | K as a multiple of the true K | true K vs eigengap-estimated K; with vs without row normalisation; normalised vs unnormalised Laplacian |

### What gets recorded (one row per run)

| Column | Meaning |
|---|---|
| `nmi`, `ari` | Agreement with the true communities (1 = perfect, 0 = random) |
| `modularity` | Modularity of the found partition |
| `modularity_truth` | Modularity of the *true* partition, for comparison |
| `n_found`, `n_true` | Number of communities found vs actual |
| `frac_disconnected` | Share of found communities that are internally split into pieces |
| `runtime_s` | Seconds spent inside `detect()` only |
| `status` | `ok`, `out_of_memory` or the error message |

`stability_<algo>.csv` stores how much the 5 seeded runs agree **with each other** (NMI between runs).

### Figures (`python make_figures.py`)

| Figure | Shows |
|---|---|
| `fig1_accuracy_vs_mu` | NMI and ARI as difficulty rises: who breaks first |
| `fig2_runtime_vs_n` | Runtime from 1k to 50k nodes (log-log) |
| `fig3_stability` | Whether different seeds give the same answer |
| `fig4_modularity_gap` | Whether the algorithm's objective still matches the truth at high μ |
| `fig5a/b_parameter_sweep` | Sensitivity to γ, Markov time and K: accuracy (a) and over- or under-splitting (b) |
| `fig6_email_bridge` | Email result placed on the LFR curve: does LFR predict reality? |
| `fig7_ablation` | Effect of switching key components on or off |
| `fig8_disconnected` | Badly connected communities (Louvain vs Leiden) |
| `fig9_recovery_by_size_*` | Which true communities get recovered, by size |
| `fig10_email_confusion` | Which departments get merged or split |
| `part1_louvain_leiden` … `part4_spectral` | One figure per report part: (a) that method vs the others, (b) its parameter sweep, (c) its ablation. Used in each person's section 5 write-up |

Plus `results/summary_table.csv` (means per graph × algorithm) and `results/failures.csv` if anything crashed.

---

## Rules so our results stay comparable

1. **Never** read `labels.txt` inside `detect()`.
2. Use `seed` for **all** randomness.
3. Don't edit `harness/core.py` or the data. If something needs changing, ask Nithish so it changes for everyone.
4. Run `smoke` before anything else.
5. Commit your `results/raw/*.csv` files; don't paste numbers into chat.
6. To change your algorithm's settings, edit only `PARAMS`, `SWEEP` and `VARIANTS` in your own file, and explain the choice in your report section.

## Regenerating the data

```bash
pip install networkit && python scripts/generate_lfr.py   # identical files every time
python scripts/prepare_email.py
```
