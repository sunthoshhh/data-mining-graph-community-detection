"""
The shared evaluation harness. Nobody needs to edit this file.

It does everything that must be IDENTICAL for all four algorithms:
loading graphs, timing, scoring, repeating with seeds, and saving results.
Each algorithm only provides a detect() function (see algorithms/).
"""

import csv
import importlib
import itertools
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback

import igraph as ig
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

# ---------------------------------------------------------------- datasets --

LFR_GRAPHS = [
    "mix_mu0.2_s1", "mix_mu0.2_s2",
    "mix_mu0.4_s1", "mix_mu0.4_s2",
    "mix_mu0.6_s1", "mix_mu0.6_s2",
    "scale_n10000", "scale_n50000",
]
ALL_GRAPHS = LFR_GRAPHS + ["email"]

SEEDS = [0, 1, 2, 3, 4]          # algorithm seeds: every graph is run 5 times
SWEEP_GRAPHS = ["mix_mu0.4_s1", "email"]
ABLATION_GRAPHS = ["mix_mu0.2_s1", "mix_mu0.4_s1", "mix_mu0.6_s1", "email"]


def graph_dir(name):
    return os.path.join(DATA, "email") if name == "email" else os.path.join(DATA, "lfr", name)


def graph_info(name):
    """Experiment tags used for grouping results and plotting."""
    if name == "email":
        return dict(dataset="email", experiment="real", mu_target="", graph_seed="")
    if name.startswith("mix"):
        mu, s = name.split("_")[1:]
        return dict(dataset="lfr", experiment="mixing", mu_target=float(mu[2:]), graph_seed=int(s[1:]))
    return dict(dataset="lfr", experiment="scale", mu_target=0.4, graph_seed=1)


def ensure_email():
    if not os.path.exists(os.path.join(DATA, "email", "edges.txt")):
        print("email-Eu-core not found, preparing it now (needs internet, one time only)...")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "prepare_email.py")], check=True)


_cache = {}


def load(name):
    """Return (igraph.Graph, truth) where truth[i] is node i's true community."""
    if name not in _cache:
        if name == "email":
            ensure_email()
        d = graph_dir(name)
        with open(os.path.join(d, "labels.txt")) as f:
            pairs = [tuple(map(int, line.split())) for line in f if line.strip()]
        truth = [c for _, c in sorted(pairs)]
        with open(os.path.join(d, "edges.txt")) as f:
            edges = [tuple(map(int, line.split())) for line in f if line.strip()]
        g = ig.Graph(n=len(truth), edges=edges, directed=False)
        _cache[name] = (g, truth)
    return _cache[name]


# ----------------------------------------------------------------- metrics --

def to_membership(result, n):
    """Accept either a membership list [c0, c1, ...] or a list of node sets."""
    result = list(result)
    if len(result) == n and all(isinstance(x, (int,)) or hasattr(x, "__index__") for x in result):
        return [int(x) for x in result]
    membership = [-1] * n
    for cid, comm in enumerate(result):
        for v in comm:
            membership[v] = cid
    if -1 in membership:
        raise ValueError("detect() left some nodes without a community")
    return membership


def frac_disconnected(g, membership):
    """Share of found communities (size > 1) that are internally disconnected."""
    groups = {}
    for v, c in enumerate(membership):
        groups.setdefault(c, []).append(v)
    big = [vs for vs in groups.values() if len(vs) > 1]
    if not big:
        return 0.0
    bad = sum(1 for vs in big if not g.subgraph(vs).is_connected())
    return bad / len(big)


def score(g, truth, membership):
    return dict(
        nmi=normalized_mutual_info_score(truth, membership),
        ari=adjusted_rand_score(truth, membership),
        modularity=g.modularity(membership),
        modularity_truth=g.modularity(truth),
        n_found=len(set(membership)),
        n_true=len(set(truth)),
        frac_disconnected=frac_disconnected(g, membership),
    )


# ------------------------------------------------------------------ runner --

def load_algorithm(name):
    return importlib.import_module(f"algorithms.{name}")


TIMEOUT_S = int(os.environ.get("CD_TIMEOUT", 900))   # per run; change with --timeout=SECONDS


def _child(conn, algo, g, seed, params):
    """Runs inside a separate process so a slow or crashing run can be stopped cleanly."""
    try:
        t0 = time.perf_counter()
        result = algo.detect(g, seed, **params)
        runtime = time.perf_counter() - t0          # only the algorithm is timed
        conn.send(("ok", to_membership(result, g.vcount()), runtime))
    except MemoryError:
        conn.send(("out_of_memory", None, None))
    except Exception as e:
        traceback.print_exc()
        conn.send((f"error: {type(e).__name__}: {e}", None, None))


def run_one(algo, graph, seed, params):
    g, truth = load(graph)
    if getattr(algo, "NEEDS_K", False):
        # methods that must be told K (spectral clustering) get the true count,
        # scaled by k_factor. Declared openly in the report as an advantage.
        params = {**params, "n_true": len(set(truth))}
    row = dict(algorithm=algo.NAME, graph=graph, **graph_info(graph), n=g.vcount(),
               edges=g.ecount(), seed=seed,
               params=_fmt({k: v for k, v in params.items() if k != "n_true"}))
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(child, algo, g, seed, params))
    p.start()
    child.close()
    membership = None
    if parent.poll(TIMEOUT_S):
        try:
            status, membership, runtime = parent.recv()
        except EOFError:           # child died without reporting, usually the OS killing it for memory
            status = "killed (likely out of memory)"
    else:
        p.kill()
        status = f"timeout (> {TIMEOUT_S}s)"
    p.join()
    row["status"] = status
    if status == "ok":
        row["runtime_s"] = runtime
        row.update(score(g, truth, membership))
    return row, membership


def _fmt(params):
    return ";".join(f"{k}={v}" for k, v in sorted(params.items())) or "default"


def _save(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {len(rows)} rows -> {os.path.relpath(path, ROOT)}")


def _save_partition(algo_key, graph, seed, membership):
    d = os.path.join(RESULTS, "partitions", algo_key)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{graph}_seed{seed}.txt"), "w") as f:
        f.writelines(f"{v} {c}\n" for v, c in enumerate(membership))


def _log(row):
    if row["status"] == "ok":
        print(f"  {row['graph']:<14} seed={row['seed']} {row['params']:<36} "
              f"NMI={row['nmi']:.3f} ARI={row['ari']:.3f} found={row['n_found']:<5} "
              f"t={row['runtime_s']:.2f}s")
    else:
        print(f"  {row['graph']:<14} seed={row['seed']} {row['status']}")


# ------------------------------------------------------------- experiments --

def _available(graphs):
    """Skip (with a clear message) any graph that can't be loaded, e.g. email offline."""
    ok = []
    for g in graphs:
        try:
            load(g)
            ok.append(g)
        except Exception as e:
            print(f"  WARNING: skipping {g} ({type(e).__name__}). "
                  f"For email, run scripts/prepare_email.py with internet access.")
    return ok


def smoke(algo_key):
    """Quick check that detect() works: easy graph, one seed, NMI must be > 0.9."""
    algo = load_algorithm(algo_key)
    row, _ = run_one(algo, "mix_mu0.2_s1", 0, dict(algo.PARAMS))
    _log(row)
    ok = row["status"] == "ok" and row["nmi"] > 0.9
    print("SMOKE TEST", "PASSED" if ok else "FAILED (NMI should be > 0.9 on the easy graph)")
    return ok


def main(algo_key, graphs=ALL_GRAPHS):
    """Every graph x 5 seeds with default settings. Feeds figures 1, 2, 3, 4, 6."""
    algo = load_algorithm(algo_key)
    rows, stab = [], []
    for graph in _available(graphs):
        memberships = []
        for seed in SEEDS:
            row, m = run_one(algo, graph, seed, dict(algo.PARAMS))
            _log(row)
            rows.append(row)
            if row["status"].startswith(("timeout", "killed")):
                print("  (skipping remaining seeds for this graph, a failure here is the result)")
                break
            if m is not None:
                memberships.append(m)
                _save_partition(algo_key, graph, seed, m)
        # stability: how much do runs with different seeds agree WITH EACH OTHER?
        pair_nmi = [normalized_mutual_info_score(a, b) for a, b in itertools.combinations(memberships, 2)]
        if pair_nmi:
            stab.append(dict(algorithm=algo.NAME, graph=graph, **graph_info(graph), runs=len(memberships),
                             between_run_nmi_mean=sum(pair_nmi) / len(pair_nmi),
                             between_run_nmi_min=min(pair_nmi)))
    _save(rows, os.path.join(RESULTS, "raw", f"main_{algo_key}.csv"))
    _save(stab, os.path.join(RESULTS, "raw", f"stability_{algo_key}.csv"))


def sweep(algo_key):
    """Parameter sensitivity: each value in SWEEP on the medium LFR graph + email."""
    algo = load_algorithm(algo_key)
    if not algo.SWEEP:
        print(f"{algo.NAME} has no parameter to sweep. Discuss that in the report instead.")
        return
    rows = []
    for pname, values in algo.SWEEP.items():
        for value, graph, seed in itertools.product(values, _available(SWEEP_GRAPHS), SEEDS):
            params = {**algo.PARAMS, pname: value}
            row, _ = run_one(algo, graph, seed, params)
            row["sweep_param"], row["sweep_value"] = pname, value
            _log(row)
            rows.append(row)
    _save(rows, os.path.join(RESULTS, "raw", f"sweep_{algo_key}.csv"))


def ablation(algo_key):
    """Turn key components on/off (VARIANTS in the algorithm file)."""
    algo = load_algorithm(algo_key)
    if len(algo.VARIANTS) < 2:
        print(f"{algo.NAME} has no ablation variants defined.")
        return
    rows = []
    for vname, vparams in algo.VARIANTS.items():
        for graph, seed in itertools.product(_available(ABLATION_GRAPHS), SEEDS):
            row, _ = run_one(algo, graph, seed, {**algo.PARAMS, **vparams})
            row["variant"] = vname
            _log(row)
            rows.append(row)
    _save(rows, os.path.join(RESULTS, "raw", f"ablation_{algo_key}.csv"))
