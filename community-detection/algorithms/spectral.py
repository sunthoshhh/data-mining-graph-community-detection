"""
Spectral clustering (Ng, Jordan & Weiss, 2002). Turns community detection into K-means:
  1. Build the normalised adjacency D^-1/2 A D^-1/2 (equivalently the normalised Laplacian)
  2. Take its top-K eigenvectors: each node becomes a point in K-dimensional space
  3. Normalise each node's row to unit length
  4. Run K-means on those points; each cluster is a community
Unlike the other methods it must be TOLD the number of communities K.
Harness gives it the true K (an advantage we state in the report) times k_factor.
Implementation: scipy (sparse eigensolver) + scikit-learn K-means.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

NAME = "Spectral"
NEEDS_K = True           # harness passes n_true (number of true communities)
PARAMS = {"k_factor": 1.0, "k_mode": "oracle", "row_norm": True, "laplacian": "normalized"}
SWEEP = {"k_factor": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]}   # K = k_factor x true K
VARIANTS = {
    "default": {},                            # true K, normalised, row-normalised
    "eigengap_k": {"k_mode": "eigengap"},     # realistic: estimate K from the eigengap, no answer key
    "no_row_norm": {"row_norm": False},       # ablation: skip step 3
    "unnormalized": {"laplacian": "unnormalized"},  # ablation: plain Laplacian L = D - A
}


def _embedding(A, k, laplacian, seed):
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    v0 = np.random.default_rng(seed).random(n)          # fixed start vector = reproducible
    if laplacian == "normalized":
        d = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
        M = d @ A @ d
        vals, vecs = eigsh(M, k=k, which="LA", v0=v0)    # largest of D^-1/2 A D^-1/2
        order = np.argsort(-vals)
    else:
        L = sp.diags(deg) - A
        vals, vecs = eigsh(L, k=k, sigma=-1e-3, which="LM", v0=v0)   # smallest of L
        order = np.argsort(vals)
    return vals[order], vecs[:, order]


def _eigengap_k(A, seed, k_max=None):
    n = A.shape[0]
    k_max = k_max or min(n // 5, 150)
    vals, _ = _embedding(A, k_max, "normalized", seed)
    gaps = vals[:-1] - vals[1:]                          # eigenvalues sorted high -> low
    return int(np.argmax(gaps[1:]) + 2)                  # ignore the trivial first gap, K >= 2


def detect(g, seed, n_true, k_factor=1.0, k_mode="oracle", row_norm=True, laplacian="normalized"):
    n = g.vcount()
    edges = np.array(g.get_edgelist())
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr().astype(float)
    if k_mode == "eigengap":
        k = _eigengap_k(A, seed)
    else:
        k = max(2, int(round(k_factor * n_true)))
    k = min(k, n - 2)
    _, X = _embedding(A, k, laplacian, seed)
    if row_norm:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X).tolist()
