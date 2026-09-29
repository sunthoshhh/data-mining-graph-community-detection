"""
Generate the LFR benchmark graphs for the community detection project.

Two experiments, 8 graphs in total:
  1. Mixing sweep : n=1000, mu in {0.2, 0.4, 0.6}, 2 seeds each  -> 6 graphs
  2. Scale sweep  : mu=0.4, n in {10000, 50000}, 1 seed each     -> 2 graphs
     (the n=1000 point is reused from the mixing sweep, mu=0.4)

Each graph is saved to data/lfr/<name>/ as:
  edges.txt   one edge per line: "u v" (undirected, nodes are 0..n-1)
  labels.txt  one line per node: "node community_id"  (the true communities)
A summary of every graph is written to data/lfr/summary.csv.

We use NetworKit's LFR generator rather than networkx's, because the networkx
version overshoots the requested mu by roughly 0.1 to 0.2.

Run:   python scripts/generate_lfr.py
Needs: pip install networkit
"""

import csv
import os
import time

import networkit as nk

# Fixed LFR settings (standard values from Lancichinetti et al., 2008)
AVG_DEGREE = 20
MAX_DEGREE = 50
TAU1 = 3        # degree distribution exponent
TAU2 = 1.5      # community size distribution exponent
MIN_COMM = 20
MAX_COMM = 100

# (name, n, mu, seed)
GRAPHS = [
    ("mix_mu0.2_s1", 1000, 0.2, 1),
    ("mix_mu0.2_s2", 1000, 0.2, 2),
    ("mix_mu0.4_s1", 1000, 0.4, 1),
    ("mix_mu0.4_s2", 1000, 0.4, 2),
    ("mix_mu0.6_s1", 1000, 0.6, 1),
    ("mix_mu0.6_s2", 1000, 0.6, 2),
    ("scale_n10000", 10000, 0.4, 1),
    ("scale_n50000", 50000, 0.4, 1),
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lfr")


def build(n, mu, seed):
    nk.setNumberOfThreads(1)                  # single thread so seeds are reproducible
    nk.setSeed(seed, False)
    gen = nk.generators.LFRGenerator(n)
    gen.generatePowerlawDegreeSequence(AVG_DEGREE, MAX_DEGREE, -TAU1)
    gen.generatePowerlawCommunitySizeSequence(MIN_COMM, MAX_COMM, -TAU2)
    gen.setMu(mu)
    gen.run()
    return gen.getGraph(), gen.getPartition()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for name, n, mu, seed in GRAPHS:
        t0 = time.time()
        G, part = build(n, mu, seed)
        comm_of = part.getVector()

        edges = [(u, v) for u, v in G.iterEdges() if u != v]
        between = sum(1 for u, v in edges if comm_of[u] != comm_of[v])

        path = os.path.join(OUT_DIR, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "edges.txt"), "w") as f:
            f.writelines(f"{u} {v}\n" for u, v in edges)
        with open(os.path.join(path, "labels.txt"), "w") as f:
            f.writelines(f"{v} {comm_of[v]}\n" for v in range(n))

        row = dict(
            name=name,
            experiment="mixing" if name.startswith("mix") else "scale",
            n=n,
            edges=len(edges),
            avg_degree=round(2 * len(edges) / n, 1),
            mu_target=mu,
            mu_actual=round(between / len(edges), 3),
            communities=len(set(comm_of)),
            seed=seed,
            gen_seconds=round(time.time() - t0, 1),
        )
        rows.append(row)
        print(row)

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
