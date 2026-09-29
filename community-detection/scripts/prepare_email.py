"""
Download and clean the email-Eu-core network (SNAP) into the same format as the LFR graphs.

Source: https://snap.stanford.edu/data/email-Eu-core.html
  Emails between members of a European research institution.
  Ground truth = the department each person belongs to (42 departments).

Cleaning steps (write these in the report's Data section):
  1. Directed -> undirected (an email either way counts as one link)
  2. Remove self-loops and duplicate edges
  3. Keep only the largest connected component (isolated people have no
     structure to detect and would distort NMI/ARI)
  4. Relabel nodes 0..n-1

Output: data/email/edges.txt, data/email/labels.txt, data/email/summary.csv
Run:    python scripts/prepare_email.py     (the harness also runs this automatically)
"""

import csv
import gzip
import io
import os
import urllib.request

import igraph as ig

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data", "email")
RAW = os.path.join(OUT, "raw")
URLS = {
    "edges": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
    "labels": "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz",
}


def fetch(name):
    """Download a SNAP file once and cache it in data/email/raw/."""
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, os.path.basename(URLS[name]))
    if not os.path.exists(path):
        print(f"Downloading {URLS[name]} ...")
        urllib.request.urlretrieve(URLS[name], path)
    with gzip.open(path, "rt") as f:
        return [tuple(map(int, line.split())) for line in f if line.strip()]


def main():
    edges = fetch("edges")
    dept = dict(fetch("labels"))

    n_raw = max(max(u, v) for u, v in edges) + 1
    n_raw = max(n_raw, max(dept) + 1)
    g = ig.Graph(n=n_raw, edges=edges, directed=True)
    g.vs["orig_id"] = list(range(n_raw))
    g.vs["dept"] = [dept[i] for i in range(n_raw)]
    raw_nodes, raw_edges = g.vcount(), g.ecount()

    g.to_undirected(mode="collapse")      # steps 1 + 2 (merges u->v and v->u)
    g.simplify()                          # step 2 (self-loops, duplicates)
    g = g.connected_components().giant()  # step 3, igraph relabels 0..n-1 (step 4)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "edges.txt"), "w") as f:
        f.writelines(f"{e.source} {e.target}\n" for e in g.es)
    with open(os.path.join(OUT, "labels.txt"), "w") as f:
        f.writelines(f"{v.index} {v['dept']}\n" for v in g.vs)

    labels = g.vs["dept"]
    between = sum(1 for e in g.es if labels[e.source] != labels[e.target])
    summary = dict(
        name="email",
        raw_nodes=raw_nodes,
        raw_directed_edges=raw_edges,
        n=g.vcount(),
        edges=g.ecount(),
        avg_degree=round(2 * g.ecount() / g.vcount(), 1),
        communities=len(set(labels)),
        mu_actual=round(between / g.ecount(), 3),   # where email sits on the LFR difficulty curve
    )
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary.keys())
        w.writeheader()
        w.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
